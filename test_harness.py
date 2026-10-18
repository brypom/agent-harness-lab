"""离线验收：虚拟配置、模拟 HTTP；不读取真实 .env，不访问网络。"""

import contextlib
import copy
import io
import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import main
import tools
import harness


FAKE_KEY = "offline-dummy-key"
CONFIG = {
    "DEEPSEEK_API_KEY": FAKE_KEY,
    "DEEPSEEK_MODEL": "offline-model",
    "AGENT_SYSTEM_PROMPT": "测试系统提示。",
}
BASE_MESSAGES = [{"role": "system", "content": "测试系统提示。"}]
FIXED_TIME = datetime(2030, 1, 2, 3, 4, 5, 123456,
                      tzinfo=timezone(timedelta(hours=8)))
FIXED_TIME_TEXT = "2030-01-02T03:04:05+08:00"
PLACE = {"name": "杭州", "admin1": "浙江省", "country": "中国",
         "latitude": 30.27, "longitude": 120.15}
GEOCODING_DATA = {"results": [PLACE]}
WEATHER_DATA = {
    "current": {"time": "2030-01-02T15:00", "temperature_2m": 22.4, "weather_code": 3},
    "current_units": {"temperature_2m": "°C", "weather_code": "wmo code"},
    "timezone": "Asia/Shanghai", "utc_offset_seconds": 28800,
}


def reply(text="模拟回答。", calls=None, reason=None):
    return {"choices": [{
        "finish_reason": reason or ("tool_calls" if calls else "stop"),
        "message": {"role": "assistant", "content": text, "tool_calls": calls},
    }]}


def tool_call(expression="23*47", call_id="call_calc_1", name="calculate"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"expression": expression}),
    }}


def time_call(call_id="call_time_1", arguments="{}"):
    return {"id": call_id, "type": "function", "function": {
        "name": "get_time", "arguments": arguments,
    }}


def weather_call(city="杭州", call_id="call_weather_1"):
    return {"id": call_id, "type": "function", "function": {
        "name": "get_weather", "arguments": json.dumps({"city": city}, ensure_ascii=False),
    }}


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        # 替换整个配置来源，不读取或备份当前进程中的真实配置值。
        self.environment = CONFIG.copy()
        patch.object(main.os, "environ", self.environment).start()
        self.env_file = MagicMock()
        self.env_file.read_text.return_value = ""
        patch.object(main, "ENV_FILE", self.env_file).start()
        self.temp_directory = tempfile.TemporaryDirectory(prefix="agent-harness-test-")
        self.addCleanup(self.temp_directory.cleanup)
        self.session_root = Path(self.temp_directory.name) / "sessions"
        patch.object(main, "SESSION_ROOT", self.session_root).start()
        patch.object(harness, "now", return_value="2030-01-02T00:00:00+00:00").start()
        patch("socket.socket.connect", side_effect=AssertionError("禁止网络连接")).start()
        self.clock = patch.object(tools, "_local_now", return_value=FIXED_TIME).start()
        self.requests = []
        self.script = []
        self.transport = patch.object(main, "urlopen", side_effect=self.respond).start()
        self.weather_requests = []
        self.weather_script = []
        self.weather_transport = patch.object(tools, "urlopen", side_effect=self.respond_weather).start()

    def respond_weather(self, request, timeout):
        parts = urlsplit(request.full_url)
        endpoint = f"{parts.scheme}://{parts.netloc}{parts.path}"
        self.assertIn(endpoint, (tools.GEOCODING_URL, tools.WEATHER_URL))
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertIsNone(request.data)
        self.assertEqual(timeout, 10)
        self.weather_requests.append({"endpoint": endpoint, "params": parse_qs(parts.query)})
        if not self.weather_script:
            raise AssertionError("发生了未预期的天气服务请求")
        item = self.weather_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        response = MagicMock()
        response.__enter__.return_value = response
        raw = item if isinstance(item, bytes) else json.dumps(item, ensure_ascii=False).encode("utf-8")

        def read(limit):
            self.assertEqual(limit, tools.MAX_WEATHER_RESPONSE_BYTES + 1)
            return raw[:limit]

        response.read.side_effect = read
        return response

    def respond(self, request, timeout):
        self.assertEqual(request.full_url, main.API_URL)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_KEY)
        self.assertEqual(timeout, 30)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "offline-model")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        if "response_format" in payload:
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["max_tokens"], 2048)
            self.assertNotIn("tools", payload)
            self.assertNotIn("tool_choice", payload)
            self.assertIn("json", payload["messages"][0]["content"])
        else:
            self.assertEqual({tool["function"]["name"] for tool in payload["tools"]},
                             {"calculate", "get_time", "get_weather"})
            self.assertEqual(payload["tool_choice"], "auto")
        self.requests.append(copy.deepcopy(payload))
        if not self.script:
            raise AssertionError("发生了未预期的模型请求")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = item if isinstance(item, bytes) else json.dumps(item, ensure_ascii=False).encode("utf-8")
        return response

    def run_cli(self, inputs):
        output = io.StringIO()
        with patch("builtins.input", side_effect=inputs), contextlib.redirect_stdout(output):
            status = main.main()
        text = output.getvalue()
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn("Authorization", text)
        self.assertNotIn("Bearer ", text)
        return status, text

    def run_turn(self, question="计算 23×47", history=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main.run_turn(BASE_MESSAGES if history is None else history, question, FAKE_KEY, "offline-model")
        self.assertNotIn(FAKE_KEY, output.getvalue())
        return result, output.getvalue()


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        cases = {"23*47": 1081, "(10*3)/2": 15, "2**8": 256,
                 "7//2": 3, "7%2": 1, "-3+5": 2, "2**-2": 0.25,
                 "1.5*2": 3, " 2+3 ": 5, "10**12": 10**12}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(tools.calculate(expression)["value"], expected)

    def test_unsupported_syntax(self):
        for expression in ("x+1", "abs(-2)", "__import__('os')", "(1).__class__",
                           "[1][0]", "[x for x in [1]]", "1 < 2", "1 << 2", "a=1"):
            with self.subTest(expression=expression), self.assertRaises(tools.ToolError):
                tools.calculate(expression)

    def test_invalid_numbers_and_division(self):
        for expression in ("True+1", "False", "1j", "(-1)**0.5", "1/0", "1//0", "1%0", "1e309"):
            with self.subTest(expression=expression), self.assertRaises(tools.ToolError):
                tools.calculate(expression)

    def test_expression_and_resource_limits(self):
        for expression in ("", " " * 201, "+" * 20 + "1", "+".join(["1"] * 30),
                           "2**13", "2**-13", "1000000000001", "10**12+1", "23×47"):
            with self.subTest(expression=expression), self.assertRaises(tools.ToolError):
                tools.calculate(expression)

    def test_parameter_validation(self):
        for arguments in (None, "{bad", "[]", "{}", '{"expression":2}',
                          '{"expression":"2+3","extra":"x"}',
                          '{"expression":"2+3","expression":"4+5"}', " " * 2001):
            with self.subTest(arguments=arguments):
                self.assertFalse(tools.execute("calculate", arguments)["ok"])

    def test_unknown_tool_and_safe_errors(self):
        self.assertFalse(tools.execute("unknown", '{}')["ok"])
        result = tools.execute("calculate", '{"expression":"1/0"}')
        self.assertFalse(result["ok"])
        self.assertIn("不能除以零", result["error"])
        self.assertNotIn("Traceback", result["error"])

    def test_successful_tool_result(self):
        self.assertEqual(tools.execute("calculate", '{"expression":"23*47"}'),
                         {"ok": True, "result": {"value": 1081}})


class TimeToolTests(unittest.TestCase):
    def test_schema_has_no_parameters(self):
        schemas = {tool["function"]["name"]: tool["function"] for tool in tools.SCHEMAS}
        self.assertEqual(set(schemas), {"calculate", "get_time", "get_weather"})
        self.assertEqual(schemas["get_time"]["parameters"], {
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        })

    def test_fixed_clock_result_has_offset_and_seconds(self):
        with patch.object(tools, "_local_now", return_value=FIXED_TIME) as clock:
            result = tools.execute("get_time", "{}")
        clock.assert_called_once_with()
        self.assertEqual(result, {"ok": True, "result": {"local_time": FIXED_TIME_TEXT}})

    def test_offsets_are_not_hardcoded(self):
        for minutes in (0, -240, 330):
            fixed = datetime(2031, 6, 7, 8, 9, 10,
                             tzinfo=timezone(timedelta(minutes=minutes)))
            with self.subTest(minutes=minutes), patch.object(tools, "_local_now", return_value=fixed):
                actual = datetime.fromisoformat(tools.get_time()["local_time"])
                self.assertEqual(actual, fixed)
                self.assertEqual(actual.utcoffset(), timedelta(minutes=minutes))

    def test_each_call_reads_clock_again(self):
        later = FIXED_TIME + timedelta(seconds=7)
        with patch.object(tools, "_local_now", side_effect=[FIXED_TIME, later]) as clock:
            first = tools.get_time()
            second = tools.get_time()
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(first["local_time"], FIXED_TIME_TEXT)
        self.assertEqual(second["local_time"], "2030-01-02T03:04:12+08:00")

    def test_extra_arguments_do_not_read_clock(self):
        for arguments in ('{"city":"北京"}', '{"timezone":"Asia/Shanghai"}',
                          '{"expression":"23*47"}', '{"extra":null}'):
            with self.subTest(arguments=arguments), patch.object(tools, "_local_now") as clock:
                result = tools.execute("get_time", arguments)
                self.assertFalse(result["ok"])
                self.assertIn("不接受任何参数", result["error"])
                clock.assert_not_called()

    def test_invalid_arguments_do_not_read_clock(self):
        for arguments in (None, {}, "", "{bad", "[]", "null", '"text"', "1",
                          '{"city":"a","city":"b"}', " " * 2001):
            with self.subTest(arguments=arguments), patch.object(tools, "_local_now") as clock:
                self.assertFalse(tools.execute("get_time", arguments)["ok"])
                clock.assert_not_called()

    def test_clock_failure_returns_safe_error(self):
        with patch.object(tools, "_local_now", side_effect=OSError("内部模拟信息")):
            result = tools.execute("get_time", "{}")
        self.assertEqual(result, {"ok": False, "error_type": "ToolError",
                                  "error": "工具未能完成请求。"})

    def test_time_without_offset_is_rejected(self):
        with patch.object(tools, "_local_now", return_value=datetime(2030, 1, 2)):
            result = tools.execute("get_time", "{}")
        self.assertFalse(result["ok"])
        self.assertIn("UTC 偏移", result["error"])


class TimeProtocolTests(OfflineCase):
    def test_time_closed_loop(self):
        self.script = [reply(None, [time_call()]), reply("本机时间：" + FIXED_TIME_TEXT)]
        (answer, history), output = self.run_turn("本机现在几点？")
        self.assertEqual(len(self.requests), 2)
        sent = self.requests[1]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "tool"])
        self.assertEqual(sent[-2]["tool_calls"][0]["function"]["arguments"], "{}")
        self.assertEqual(sent[-1]["tool_call_id"], "call_time_1")
        self.assertEqual(json.loads(sent[-1]["content"]),
                         {"ok": True, "result": {"local_time": FIXED_TIME_TEXT}})
        self.assertEqual(history[-1]["content"], answer)
        self.assertIn(FIXED_TIME_TEXT, output)
        self.clock.assert_called_once_with()

    def test_time_and_calculate_in_same_batch(self):
        self.script = [reply(None, [time_call("time_a"), tool_call(call_id="calc_b")]), reply()]
        self.run_turn("查询本机时间，并计算 23×47")
        results = self.requests[1]["messages"][-2:]
        self.assertEqual([m["tool_call_id"] for m in results], ["time_a", "calc_b"])
        self.assertEqual(json.loads(results[0]["content"])["result"]["local_time"], FIXED_TIME_TEXT)
        self.assertEqual(json.loads(results[1]["content"])["result"]["value"], 1081)

    def test_extra_parameters_return_error_with_original_id(self):
        self.script = [reply(None, [time_call(arguments='{"city":"北京"}')]),
                       reply("时间工具不支持城市查询。")]
        self.run_turn("北京几点？")
        result = self.requests[1]["messages"][-1]
        self.assertEqual(result["tool_call_id"], "call_time_1")
        self.assertFalse(json.loads(result["content"])["ok"])
        self.clock.assert_not_called()

    def test_clock_error_returns_to_model(self):
        self.clock.side_effect = OSError("内部模拟信息")
        self.script = [reply(None, [time_call()]), reply("读取本机时间失败。")]
        (answer, history), output = self.run_turn("本机几点？")
        result = self.requests[1]["messages"][-1]
        self.assertFalse(json.loads(result["content"])["ok"])
        self.assertNotIn("内部模拟信息", output)
        self.assertEqual(answer, "读取本机时间失败。")

    def test_failed_time_turn_is_absent_from_next_question(self):
        self.script = [reply("旧回答"), reply(None, [time_call()]), TimeoutError(), reply("新回答")]
        status, output = self.run_cli(["旧问题", "本机几点？", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("本轮未加入历史", output)
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]],
                         ["旧问题", "旧回答", "新问题"])

    def test_two_time_turns_read_new_times_and_keep_history(self):
        self.clock.side_effect = [FIXED_TIME, FIXED_TIME + timedelta(seconds=7)]
        self.script = [reply(None, [time_call("time_a")]), reply("第一次时间"),
                       reply(None, [time_call("time_b")]), reply("第二次时间")]
        status, output = self.run_cli(["本机现在几点？", "现在再查一次", "exit"])
        self.assertEqual(status, 0)
        sent = self.requests[-1]["messages"]
        results = [json.loads(m["content"])["result"]["local_time"]
                   for m in sent if m["role"] == "tool"]
        self.assertEqual(results, [FIXED_TIME_TEXT, "2030-01-02T03:04:12+08:00"])
        self.assertIn("get_time", sent[0]["content"])
        self.assertEqual(self.clock.call_count, 2)


class WeatherToolTests(OfflineCase):
    def query(self, city="杭州"):
        return tools.execute("get_weather", json.dumps({"city": city}, ensure_ascii=False))

    def test_unique_place_queries_weather_and_retains_metadata(self):
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA]
        result = self.query()
        self.assertEqual(result, {"ok": True, "result": {
            "needs_clarification": False, "location": PLACE,
            "current": {**WEATHER_DATA["current"], "weather_description": "阴天"},
            "units": {"temperature_2m": "°C"}, "timezone": "Asia/Shanghai",
            "utc_offset_seconds": 28800, "source": "Open-Meteo",
        }})
        self.assertEqual(self.weather_requests, [
            {"endpoint": tools.GEOCODING_URL,
             "params": {"name": ["杭州"], "count": ["5"], "language": ["zh"]}},
            {"endpoint": tools.WEATHER_URL, "params": {
                "latitude": ["30.27"], "longitude": ["120.15"],
                "current": ["temperature_2m,weather_code"],
                "temperature_unit": ["celsius"], "timezone": ["auto"],
            }},
        ])
        self.assertFalse(self.weather_script)
        self.clock.assert_not_called()
        self.transport.assert_not_called()

    def test_ambiguous_place_returns_candidates_without_weather(self):
        other = {**PLACE, "admin1": "另一个省份"}
        self.weather_script = [{"results": [PLACE, other]}]
        result = self.query()
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], {"needs_clarification": True, "candidates": [
            {"name": "杭州", "admin1": "浙江省", "country": "中国"},
            {"name": "杭州", "admin1": "另一个省份", "country": "中国"},
        ]})
        self.assertEqual(len(self.weather_requests), 1)

    def test_missing_place_returns_error(self):
        for data in ({}, {"results": []}):
            with self.subTest(data=data):
                self.weather_script = [data]
                result = self.query()
                self.assertFalse(result["ok"])
                self.assertIn("未找到地点", result["error"])
        self.assertEqual(self.weather_transport.call_count, 2)

    def test_invalid_parameters_do_not_request(self):
        arguments = [None, "{bad", "[]", "null", "{}",
                     '{"city":"杭州","date":"2030-01-02"}',
                     '{"city":"杭州","city":"北京"}', " " * 2001]
        arguments += [json.dumps({"city": value}) for value in
                      (None, 1, True, [], {}, "", "  ", "x" * 81)]
        for argument in arguments:
            with self.subTest(argument=argument):
                self.assertFalse(tools.execute("get_weather", argument)["ok"])
        self.weather_transport.assert_not_called()

    def test_qualified_name_and_query_characters_are_encoded(self):
        city = "  Paris, France &测试=1  "
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA]
        self.assertTrue(self.query(city)["ok"])
        params = self.weather_requests[0]["params"]
        self.assertEqual(params["name"], [city.strip()])
        self.assertNotIn("测试", params)

    def test_invalid_candidate_responses_stop_before_weather(self):
        invalid = [{"results": None}, {"results": {}}, {"results": [None]},
                   {"results": [{**PLACE, "name": ""}]},
                   {"results": [{**PLACE, "country": []}]},
                   {"results": [PLACE] * 6}]
        for data in invalid:
            with self.subTest(data=data):
                before = len(self.weather_requests)
                self.weather_script = [data]
                self.assertFalse(self.query()["ok"])
                self.assertEqual(len(self.weather_requests), before + 1)

    def test_invalid_coordinates_stop_before_weather(self):
        for field, value in (("latitude", None), ("latitude", True), ("latitude", "30"),
                             ("latitude", 91), ("latitude", float("nan")),
                             ("longitude", -181), ("longitude", float("inf"))):
            with self.subTest(field=field, value=value):
                before = len(self.weather_requests)
                self.weather_script = [{"results": [{**PLACE, field: value}]}]
                result = self.query()
                self.assertFalse(result["ok"])
                self.assertIn("经纬度", result["error"])
                self.assertEqual(len(self.weather_requests), before + 1)

    def test_missing_weather_fields_are_rejected(self):
        cases = []
        for field in ("current", "current_units", "timezone", "utc_offset_seconds"):
            data = copy.deepcopy(WEATHER_DATA)
            del data[field]
            cases.append(data)
        for field in ("time", "temperature_2m", "weather_code"):
            data = copy.deepcopy(WEATHER_DATA)
            del data["current"][field]
            cases.append(data)
        for data in cases:
            with self.subTest(data=data):
                self.weather_script = [GEOCODING_DATA, data]
                self.assertFalse(self.query()["ok"])

    def test_invalid_weather_fields_are_rejected(self):
        cases = [("current", "time", "2030-01-02"), ("current", "time", "not-a-timeT"),
                 ("current", "time", None), ("current", "temperature_2m", True),
                 ("current", "temperature_2m", "22.4"),
                 ("current", "temperature_2m", float("nan")),
                 ("current", "temperature_2m", float("inf")),
                 ("current", "weather_code", True), ("current", "weather_code", 3.5),
                 ("current", "weather_code", -1),
                 ("current_units", "temperature_2m", "°F"),
                 ("current_units", "temperature_2m", None),
                 (None, "timezone", ""), (None, "timezone", []),
                 (None, "utc_offset_seconds", True), (None, "utc_offset_seconds", 86400)]
        for section, field, value in cases:
            with self.subTest(section=section, field=field, value=value):
                data = copy.deepcopy(WEATHER_DATA)
                target = data if section is None else data[section]
                target[field] = value
                self.weather_script = [GEOCODING_DATA, data]
                self.assertFalse(self.query()["ok"])

    def test_unknown_code_is_preserved_without_guessing(self):
        data = copy.deepcopy(WEATHER_DATA)
        data["current"]["weather_code"] = 123
        self.weather_script = [GEOCODING_DATA, data]
        result = self.query()
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["current"]["weather_code"], 123)
        self.assertEqual(result["result"]["current"]["weather_description"], "未知天气编码（123）")

    def test_remote_errors_are_safe_and_not_retried(self):
        errors = [TimeoutError(FAKE_KEY), URLError(FAKE_KEY), OSError(FAKE_KEY),
                  HTTPError(tools.GEOCODING_URL, 429, FAKE_KEY, {}, io.BytesIO()),
                  HTTPError(tools.GEOCODING_URL, 500, FAKE_KEY, {}, io.BytesIO())]
        for error in errors:
            for phase in ("geocoding", "weather"):
                with self.subTest(error=type(error).__name__, phase=phase):
                    before = len(self.weather_requests)
                    self.weather_script = [error] if phase == "geocoding" else [GEOCODING_DATA, error]
                    result = self.query()
                    self.assertFalse(result["ok"])
                    self.assertNotIn(FAKE_KEY, json.dumps(result))
                    self.assertEqual(len(self.weather_requests) - before, 1 if phase == "geocoding" else 2)
                    self.assertFalse(self.weather_script)

    def test_invalid_json_and_service_error_are_rejected(self):
        cases = [b"{bad", b"\xff", [], None, {"error": True, "reason": FAKE_KEY}]
        for data in cases:
            for phase in ("geocoding", "weather"):
                with self.subTest(data=data, phase=phase):
                    self.weather_script = [data] if phase == "geocoding" else [GEOCODING_DATA, data]
                    result = self.query()
                    self.assertFalse(result["ok"])
                    self.assertNotIn(FAKE_KEY, json.dumps(result))

    def test_oversized_response_is_rejected(self):
        huge = b" " * (tools.MAX_WEATHER_RESPONSE_BYTES + 1)
        for script in ([huge], [GEOCODING_DATA, huge]):
            with self.subTest(phase=len(script)):
                self.weather_script = script.copy()
                result = self.query()
                self.assertFalse(result["ok"])
                self.assertIn("响应过大", result["error"])

    def test_fetch_helper_rejects_other_endpoints(self):
        with self.assertRaises(tools.ToolError):
            tools._fetch_json("https://example.com", {"name": "杭州"})
        self.weather_transport.assert_not_called()


class WeatherProtocolTests(OfflineCase):
    def test_weather_closed_loop_keeps_id_and_result(self):
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA]
        self.script = [reply(None, [weather_call()]), reply("模拟天气回答。")]
        (answer, history), output = self.run_turn("杭州当前天气如何？")
        self.assertEqual(answer, "模拟天气回答。")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.weather_requests), 2)
        result = self.requests[-1]["messages"][-1]
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call_weather_1")
        self.assertTrue(json.loads(result["content"])["ok"])
        self.assertEqual(history[-1]["role"], "assistant")
        self.assertIn("Open-Meteo", output)

    def test_clarification_then_new_query(self):
        other = {**PLACE, "admin1": "另一个省份"}
        self.weather_script = [{"results": [PLACE, other]}, GEOCODING_DATA, WEATHER_DATA]
        self.script = [reply(None, [weather_call()]), reply("请确认省份。"),
                       reply(None, [weather_call("杭州, 浙江省")]), reply("模拟天气回答。")]
        status, output = self.run_cli(["杭州天气", "浙江省的杭州", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual([r["endpoint"] for r in self.weather_requests],
                         [tools.GEOCODING_URL, tools.GEOCODING_URL, tools.WEATHER_URL])
        self.assertEqual(self.weather_requests[1]["params"]["name"], ["杭州, 浙江省"])
        messages = self.requests[-1]["messages"]
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        self.assertTrue(results[0]["result"]["needs_clarification"])
        self.assertFalse(results[1]["result"]["needs_clarification"])

    def test_three_tools_in_same_batch(self):
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA]
        self.script = [reply(None, [time_call("a"), weather_call(call_id="b"),
                                   tool_call(call_id="c")]), reply()]
        self.run_turn("查询本机时间、杭州天气并计算 23×47")
        results = self.requests[-1]["messages"][-3:]
        self.assertEqual([m["tool_call_id"] for m in results], ["a", "b", "c"])
        self.assertEqual(json.loads(results[0]["content"])["result"]["local_time"], FIXED_TIME_TEXT)
        self.assertEqual(json.loads(results[1]["content"])["result"]["source"], "Open-Meteo")
        self.assertEqual(json.loads(results[2]["content"])["result"]["value"], 1081)

    def test_weather_error_returns_with_original_id(self):
        self.weather_script = [TimeoutError(FAKE_KEY)]
        self.script = [reply(None, [weather_call()]), reply("查询失败，请稍后重试。")]
        (answer, history), output = self.run_turn("杭州天气如何？")
        result = self.requests[-1]["messages"][-1]
        self.assertEqual(result["tool_call_id"], "call_weather_1")
        self.assertFalse(json.loads(result["content"])["ok"])
        self.assertIn("查询失败", answer)
        self.assertNotIn(FAKE_KEY, output)

    def test_model_failure_after_weather_preserves_history(self):
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA]
        self.script = [reply("旧回答"), reply(None, [weather_call()]), TimeoutError(), reply("新回答")]
        status, output = self.run_cli(["旧问题", "杭州天气如何？", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("本轮未加入历史", output)
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]],
                         ["旧问题", "旧回答", "新问题"])

    def test_repeated_weather_query_fetches_again(self):
        newer = copy.deepcopy(WEATHER_DATA)
        newer["current"].update(time="2030-01-02T15:15", temperature_2m=23.0)
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA, GEOCODING_DATA, newer]
        self.script = [reply(None, [weather_call()]), reply("第一次天气"),
                       reply(None, [weather_call()]), reply("第二次天气")]
        status, output = self.run_cli(["杭州天气", "再查一次当前天气", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.weather_requests), 4)
        sent = self.requests[-1]["messages"]
        times = [json.loads(m["content"])["result"]["current"]["time"]
                 for m in sent if m["role"] == "tool"]
        self.assertEqual(times, ["2030-01-02T15:00", "2030-01-02T15:15"])
        self.assertIn("不支持未来或历史天气", sent[0]["content"])


class ProtocolTests(OfflineCase):
    def test_direct_answer_uses_one_model_request(self):
        self.script = [reply("你好。")]
        (answer, history), output = self.run_turn("你好")
        self.assertEqual(answer, "你好。")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant"])
        self.assertNotIn("工具调用", output)

    def test_calculation_closed_loop(self):
        self.script = [reply(None, [tool_call()]), reply("23×47＝1081。")]
        original = copy.deepcopy(BASE_MESSAGES)
        (answer, history), output = self.run_turn(history=original)
        self.assertEqual(original, BASE_MESSAGES)
        self.assertEqual(answer, "23×47＝1081。")
        self.assertEqual(len(self.requests), 2)
        sent = self.requests[1]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "tool"])
        self.assertEqual(sent[-1]["tool_call_id"], sent[-2]["tool_calls"][0]["id"])
        self.assertEqual(json.loads(sent[-1]["content"])["result"]["value"], 1081)
        self.assertEqual(history[-1]["role"], "assistant")
        self.assertIn("工具结果", output)
        self.assertIn("1081", output)

    def test_multiple_calculations_keep_id_and_order(self):
        self.script = [reply(None, [tool_call("2+3", "a"), tool_call("6*7", "b")]), reply("5 和 42。")]
        self.run_turn("请计算两个表达式")
        results = self.requests[1]["messages"][-2:]
        self.assertEqual([r["tool_call_id"] for r in results], ["a", "b"])
        self.assertEqual([json.loads(r["content"])["result"]["value"] for r in results], [5, 42])

    def test_tool_errors_return_with_original_id(self):
        calls = [tool_call("1/0", "a"), tool_call("2+3", "b", "unknown")]
        calls.append(tool_call("2+3", "c"))
        calls[-1]["function"]["arguments"] = "{bad"
        self.script = [reply(None, calls), reply("计算未成功。")]
        self.run_turn()
        results = self.requests[1]["messages"][-3:]
        self.assertEqual([r["tool_call_id"] for r in results], ["a", "b", "c"])
        self.assertTrue(all(not json.loads(r["content"])["ok"] for r in results))

    def test_invalid_batch_executes_no_tools(self):
        bad_calls = [None, {"id": "x", "type": "other"}, tool_call(call_id=""),
                     {"id": "x", "type": "function", "function": None}]
        wrong_arguments = tool_call(call_id="x")
        wrong_arguments["function"]["arguments"] = {}
        bad_calls.append(wrong_arguments)
        for bad in bad_calls:
            with self.subTest(bad=bad):
                self.script = [reply(None, [tool_call(call_id="a"), bad])]
                with patch.object(main, "execute") as execute, self.assertRaises(main.ResponseError):
                    self.run_turn()
                execute.assert_not_called()

    def test_duplicate_ids_in_batch(self):
        self.script = [reply(None, [tool_call(), tool_call()])]
        with patch.object(main, "execute") as execute, self.assertRaises(main.ResponseError):
            self.run_turn()
        execute.assert_not_called()

    def test_duplicate_ids_between_model_requests(self):
        self.script = [reply(None, [tool_call()]), reply(None, [tool_call()])]
        with patch.object(main, "execute", wraps=tools.execute) as execute, self.assertRaises(main.ResponseError):
            self.run_turn()
        self.assertEqual(execute.call_count, 1)

    def test_tool_batch_limit(self):
        self.script = [reply(None, [tool_call(call_id=str(i)) for i in range(9)])]
        with patch.object(main, "execute") as execute, self.assertRaises(main.ResponseError):
            self.run_turn()
        execute.assert_not_called()

    def test_model_request_limit_skips_last_batch(self):
        self.script = [reply(None, [tool_call(call_id=str(i))]) for i in range(6)]
        with patch.object(main, "execute", wraps=tools.execute) as execute, self.assertRaises(main.ResponseError):
            self.run_turn()
        self.assertEqual(len(self.requests), 6)
        self.assertEqual(execute.call_count, 5)

    def test_failure_after_tool_preserves_original_history(self):
        old = BASE_MESSAGES + [{"role": "user", "content": "旧问题"},
                               {"role": "assistant", "content": "旧回答"}]
        before = copy.deepcopy(old)
        self.script = [reply(None, [tool_call()]), TimeoutError()]
        with self.assertRaises(TimeoutError):
            self.run_turn(history=old)
        self.assertEqual(old, before)

    def test_invalid_final_answer_after_tool_does_not_commit(self):
        old = copy.deepcopy(BASE_MESSAGES)
        self.script = [reply(None, [tool_call()]), reply("")]
        with self.assertRaises(main.ResponseError):
            self.run_turn(history=old)
        self.assertEqual(old, BASE_MESSAGES)

    def test_malformed_responses_are_rejected(self):
        cases = [b"{bad", b"\xff", None, [], {}, {"choices": []}, reply(""),
                 reply("partial", reason="length"), reply("blocked", reason="content_filter"),
                 reply(None, [tool_call()], reason="stop"), reply("text", reason="tool_calls")]
        wrong_role = reply()
        wrong_role["choices"][0]["message"]["role"] = "user"
        cases.append(wrong_role)
        for response in cases:
            with self.subTest(response=response):
                self.script = [response]
                with self.assertRaises((main.ResponseError, json.JSONDecodeError, UnicodeDecodeError)):
                    self.run_turn()

    def test_oversized_response(self):
        self.script = [b" " * (main.MAX_RESPONSE_BYTES + 1)]
        with self.assertRaises(main.ResponseError):
            self.run_turn()

    def test_optional_protocol_field_is_preserved_not_printed(self):
        response = reply(None, [tool_call()])
        response["choices"][0]["message"]["reasoning_content"] = "模拟协议字段"
        self.script = [response, reply()]
        (answer, history), output = self.run_turn()
        self.assertEqual(self.requests[1]["messages"][-2]["reasoning_content"], "模拟协议字段")
        self.assertNotIn("模拟协议字段", output)


class ConversationTests(OfflineCase):
    def test_five_turn_history(self):
        self.script = [reply("答" + str(i)) for i in range(5)]
        status, output = self.run_cli(["第一问", "第二问", "第三问", "第四问", "第五问", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 5)
        sent = self.requests[4]["messages"]
        self.assertEqual(len(sent), 10)
        self.assertEqual(sent[1]["content"], "第一问")
        self.assertEqual(sent[2]["content"], "答0")

    def test_failed_tool_turn_is_absent_from_next_question(self):
        self.script = [reply("旧回答"), reply(None, [tool_call()]), TimeoutError(), reply("新回答")]
        status, output = self.run_cli(["旧问题", "失败问题", "新问题", "quit"])
        self.assertEqual(status, 0)
        self.assertIn("本轮未加入历史", output)
        sent = self.requests[-1]["messages"]
        self.assertEqual([m["content"] for m in sent[1:]], ["旧问题", "旧回答", "新问题"])

    def test_successful_tool_turn_is_present_in_next_question(self):
        self.script = [reply(None, [tool_call()]), reply("1081"), reply("前面算过了。")]
        status, output = self.run_cli(["计算 23×47", "刚才结果是什么？", "exit"])
        self.assertEqual(status, 0)
        sent = self.requests[-1]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "tool", "assistant", "user"])

    def test_blank_and_exit_inputs_do_not_request(self):
        for command in ("exit", "quit", " EXIT "):
            with self.subTest(command=command):
                status, output = self.run_cli(["", "  ", command])
                self.assertEqual(status, 0)
        self.transport.assert_not_called()

    def test_eof_and_cancel_stop(self):
        for error, expected in ((EOFError(), 0), (KeyboardInterrupt(), 130)):
            with self.subTest(error=type(error).__name__):
                status, output = self.run_cli([error])
                self.assertEqual(status, expected)
        self.transport.assert_not_called()

    def test_cancel_during_tool_turn(self):
        self.script = [reply(None, [tool_call()]), KeyboardInterrupt()]
        status, output = self.run_cli(["计算 23×47"])
        self.assertEqual(status, 130)
        self.assertIn("本轮未加入历史", output)

    def test_restart_has_no_old_history(self):
        self.script = [reply()]
        self.run_cli(["旧问题", "exit"])
        self.script = [reply()]
        self.run_cli(["新问题", "exit"])
        self.assertEqual(len(self.requests[-1]["messages"]), 2)

    def test_external_errors_are_safe_and_loop_continues(self):
        errors = [TimeoutError(FAKE_KEY), URLError(FAKE_KEY), ValueError(FAKE_KEY),
                  HTTPError(main.API_URL, 401, FAKE_KEY, {"Authorization": FAKE_KEY}, io.BytesIO(FAKE_KEY.encode()))]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.script = [error, reply("恢复成功。")]
                status, output = self.run_cli(["失败问题", "恢复问题", "exit"])
                self.assertEqual(status, 0)
                self.assertIn("恢复成功。", output)
                self.assertEqual(len(self.requests[-1]["messages"]), 2)


class SessionTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.store = harness.Store(self.session_root)

    def ids(self, output):
        return re.findall(r"Session ID：(ses_[0-9a-f]{12})", output)

    def read(self, sid):
        return json.loads(self.store.path(sid).read_text(encoding="utf-8"))

    def turns(self, session):
        return session["threads"][session["active_thread"]]["turns"]

    def completed_session(self, prompt="保存的系统提示。"):
        session = self.store.create(prompt)
        harness.run_saved_turn(session, self.store, "旧问题", lambda messages: (
            "旧回答", messages + [{"role": "user", "content": "旧问题"},
                                   {"role": "assistant", "content": "旧回答"}]))
        return session

    def test_create_roundtrip_and_context_is_independent(self):
        session = self.completed_session()
        loaded = harness.Store(self.session_root).load(session["id"])
        self.assertEqual(loaded, session)
        messages = harness.context(loaded)
        self.assertEqual([m["content"] for m in messages], ["保存的系统提示。", "旧问题", "旧回答"])
        messages[1]["content"] = "修改副本"
        self.assertEqual(self.turns(loaded)[0]["messages"][0]["content"], "旧问题")
        self.assertNotIn(FAKE_KEY, self.store.path(session["id"]).read_text(encoding="utf-8"))

    def test_restart_load_includes_saved_preference_in_request(self):
        self.script = [reply("你喜欢37。")]
        status, output = self.run_cli(["我喜欢的数字是37。", "exit"])
        sid = self.ids(output)[0]
        self.script = [reply("你喜欢37。")]
        status, output = self.run_cli(["/load " + sid, "我喜欢什么数字？", "exit"])
        self.assertEqual(status, 0)
        sent = self.requests[-1]["messages"]
        self.assertEqual([m["content"] for m in sent[1:]],
                         ["我喜欢的数字是37。", "你喜欢37。", "我喜欢什么数字？"])
        self.assertEqual(len(self.turns(self.read(sid))), 2)

    def test_restart_restores_complete_tool_protocol(self):
        self.script = [reply(None, [tool_call()]), reply("1081")]
        status, output = self.run_cli(["计算23×47", "exit"])
        sid = self.ids(output)[0]
        saved = self.read(sid)
        self.assertEqual([m["role"] for m in self.turns(saved)[0]["messages"]],
                         ["user", "assistant", "tool", "assistant"])
        self.script = [reply("之前结果是1081。")]
        self.run_cli(["/load " + sid, "刚才结果？", "exit"])
        sent = self.requests[-1]["messages"]
        self.assertEqual([m["role"] for m in sent],
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(sent[3]["tool_call_id"], "call_calc_1")

    def test_failed_turn_is_saved_without_exception_text_or_messages(self):
        self.script = [reply("旧回答"), reply(None, [tool_call()]), TimeoutError(FAKE_KEY)]
        status, output = self.run_cli(["旧问题", "失败问题", "exit"])
        sid = self.ids(output)[0]
        turns = self.turns(self.read(sid))
        self.assertEqual([t["status"] for t in turns], ["completed", "failed"])
        self.assertEqual(turns[-1]["input_text"], "失败问题")
        self.assertEqual(turns[-1]["messages"], [])
        self.assertEqual(turns[-1]["runs"][0]["error_type"], "TimeoutError")
        self.assertNotIn(FAKE_KEY, self.store.path(sid).read_text(encoding="utf-8"))
        self.script = [reply("新回答")]
        self.run_cli(["/load " + sid, "新问题", "exit"])
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]],
                         ["旧问题", "旧回答", "新问题"])

    def test_cancel_records_interrupted_turn(self):
        self.script = [KeyboardInterrupt()]
        status, output = self.run_cli(["取消问题"])
        self.assertEqual(status, 130)
        turn = self.turns(self.read(self.ids(output)[0]))[0]
        self.assertEqual(turn["status"], "interrupted")
        self.assertEqual(turn["messages"], [])
        self.assertEqual(turn["runs"][0]["error_type"], "KeyboardInterrupt")

    def test_loading_running_turn_marks_interrupted_without_reexecution(self):
        session = self.completed_session()
        harness.begin_turn(session, self.store, "未完成任务")
        loaded = self.store.load(session["id"])
        self.assertEqual(self.turns(loaded)[-1]["status"], "interrupted")
        self.assertEqual(self.turns(loaded)[-1]["runs"][0]["error_type"], "Interrupted")
        self.assertEqual(self.read(session["id"]), loaded)
        self.assertEqual(len(harness.context(loaded)), 3)
        self.transport.assert_not_called()
        self.weather_transport.assert_not_called()

    def test_new_session_clears_context_and_keeps_old_file(self):
        self.script = [reply("旧回答"), reply("新回答")]
        status, output = self.run_cli(["旧问题", "/new", "新问题", "exit"])
        old_id, new_id = self.ids(output)
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(len(self.requests[-1]["messages"]), 2)
        self.assertEqual(self.turns(self.read(old_id))[0]["input_text"], "旧问题")
        self.assertEqual(self.turns(self.read(new_id))[0]["input_text"], "新问题")

    def test_local_commands_and_blank_input_create_no_turns_or_requests(self):
        status, startup = self.run_cli(["/exit"])
        self.assertEqual(status, 0)
        self.assertIn("输入 help 或 /help", startup)
        self.assertNotIn("可用的本地命令", startup)
        status, output = self.run_cli(["", "help", "/help", " HELP ", "/HELP", "/session", "/load", "/load invalid",
                                       "/load a b", "/unknown", "/new extra", "/exit"])
        self.assertEqual(status, 0)
        self.assertIn("可用的本地命令", output)
        ids = self.ids(output)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(self.turns(self.read(ids[0])), [])
        self.transport.assert_not_called()
        self.weather_transport.assert_not_called()

    def test_load_failure_keeps_current_context(self):
        bad_id = "ses_000000000000"
        self.session_root.mkdir()
        self.store.path(bad_id).write_text("{bad", encoding="utf-8")
        self.script = [reply("旧回答"), reply("新回答")]
        status, output = self.run_cli(["旧问题", "/load " + bad_id, "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("加载失败", output)
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]],
                         ["旧问题", "旧回答", "新问题"])
        self.assertEqual(self.store.path(bad_id).read_text(encoding="utf-8"), "{bad")

    def test_id_restrictions_and_missing_file(self):
        for sid in ("../.env", "..\\.env", "D:\\file", "ses_123", "ses_ABCDEF123456", [], None):
            with self.subTest(sid=sid), self.assertRaises(harness.SessionError):
                self.store.load(sid)
        with self.assertRaises(harness.SessionError):
            self.store.load("ses_000000000000")
        self.assertFalse(self.session_root.exists())

    def test_inaccessible_paths_return_safe_errors(self):
        with patch.object(harness.Path, "resolve", side_effect=PermissionError(FAKE_KEY)):
            with self.assertRaises(harness.SessionError) as caught:
                self.store.load("ses_000000000000")
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            status, output = self.run_cli(["不应执行"])
        self.assertEqual(status, 1)
        self.assertIn("程序停止", output)
        self.transport.assert_not_called()

    def test_invalid_structure_version_or_identity_does_not_overwrite_file(self):
        session = self.completed_session()
        target = self.store.path(session["id"])
        cases = [[], None, {}]
        for field, value in (("version", 5), ("version", True), ("threads", []),
                             ("active_thread", "thr_000000000000"),
                             ("id", "ses_000000000000"), ("system_prompt", None)):
            data = copy.deepcopy(session)
            data[field] = value
            cases.append(data)
        secret_field = copy.deepcopy(session)
        secret_field["DEEPSEEK_API_KEY"] = FAKE_KEY
        cases.append(secret_field)
        for data in cases:
            with self.subTest(data=data):
                text = json.dumps(data)
                target.write_text(text, encoding="utf-8")
                with self.assertRaises(harness.SessionError):
                    self.store.load(session["id"])
                self.assertEqual(target.read_text(encoding="utf-8"), text)

    def test_invalid_turn_states_and_duplicate_ids_are_rejected(self):
        session = self.completed_session()
        cases = []
        for field, value in (("status", []), ("status", "failed"), ("runs", []),
                             ("input_text", ""), ("messages", [])):
            data = copy.deepcopy(session)
            self.turns(data)[0][field] = value
            cases.append(data)
        data = copy.deepcopy(session)
        self.turns(data).append(copy.deepcopy(self.turns(data)[0]))
        cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(harness.SessionError):
                harness.validate_session(data)

    def test_loaded_protocol_rejects_orphaned_tools_missing_results_and_bad_roles(self):
        self.script = [reply(None, [tool_call()]), reply("1081")]
        status, output = self.run_cli(["计算23×47", "exit"])
        sid = self.ids(output)[0]
        valid = self.read(sid)
        original = self.turns(valid)[0]["messages"]
        cases = [original[:2] + original[3:], original[:-1],
                 [{"role": "system", "content": "不应出现在轮次里"}] + original[1:]]
        wrong_id = copy.deepcopy(original)
        wrong_id[2]["tool_call_id"] = "other"
        cases.append(wrong_id)
        wrong_user = copy.deepcopy(original)
        wrong_user[0]["content"] = "其他问题"
        cases.append(wrong_user)
        for messages in cases:
            with self.subTest(messages=messages):
                data = copy.deepcopy(valid)
                self.turns(data)[0]["messages"] = messages
                self.store.path(sid).write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(harness.SessionError):
                    self.store.load(sid)

    def test_duplicate_json_fields_invalid_encoding_and_nonfinite_json(self):
        session = self.store.create("系统提示。")
        target = self.store.path(session["id"])
        for raw in (b'{"version":1,"version":1}', b"\xff", b'{"version":NaN}'):
            with self.subTest(raw=raw):
                target.write_bytes(raw)
                with self.assertRaises(harness.SessionError):
                    self.store.load(session["id"])
                self.assertEqual(target.read_bytes(), raw)

    def test_replace_or_fsync_failure_preserves_last_snapshot_and_cleans_temp(self):
        session = self.store.create("系统提示。")
        target = self.store.path(session["id"])
        old = target.read_bytes()
        session["system_prompt"] = "新提示。"
        for function in ("replace", "fsync"):
            with self.subTest(function=function), patch.object(harness.os, function,
                                                               side_effect=OSError(FAKE_KEY)):
                with self.assertRaises(harness.StorageError) as caught:
                    self.store.save(session)
                self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.assertEqual(target.read_bytes(), old)
            self.assertEqual(list(self.session_root.glob("*.tmp")), [])

    def test_cli_stops_on_start_begin_or_completion_save_failure(self):
        real_save = harness.Store.save
        # create、run.started、step.started、llm.usage、step.completed、run.completed。
        for failing_call, expected_requests in ((1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1)):
            calls = []

            def save(store, session):
                calls.append(session["id"])
                if len(calls) == failing_call:
                    raise harness.StorageError("模拟保存失败，程序停止。")
                return real_save(store, session)

            self.requests.clear()
            self.script = [reply("不能提前展示的最终回答")]
            with self.subTest(failing_call=failing_call), patch.object(harness.Store, "save", save):
                status, output = self.run_cli(["问题", "不应继续处理", "exit"])
            self.assertEqual(status, 1)
            self.assertEqual(len(self.requests), expected_requests)
            self.assertNotIn("不能提前展示的最终回答", output)
            self.assertNotIn("模型回答：", output)
            self.assertEqual(len(calls), failing_call)
            if failing_call >= 3:
                self.assertEqual(self.turns(self.read(calls[0]))[0]["status"], "running")

    def test_loading_interrupted_snapshot_save_failure_does_not_claim_restored(self):
        session = self.store.create("旧提示。")
        harness.begin_turn(session, self.store, "未完成")
        real_save = harness.Store.save

        def save(store, candidate):
            if candidate["id"] == session["id"]:
                raise harness.StorageError("模拟保存失败，程序停止。")
            real_save(store, candidate)

        with patch.object(harness.Store, "save", save):
            status, output = self.run_cli(["/load " + session["id"], "不应执行"])
        self.assertEqual(status, 1)
        self.assertNotIn("会话已恢复", output)
        self.assertEqual(self.turns(self.read(session["id"]))[0]["status"], "running")
        self.transport.assert_not_called()

    def test_restore_keeps_original_system_prompt(self):
        session = self.completed_session(prompt="原始完整系统提示。")
        self.environment["AGENT_SYSTEM_PROMPT"] = "新的配置提示。"
        self.script = [reply()]
        self.run_cli(["/load " + session["id"], "继续", "exit"])
        self.assertEqual(self.requests[-1]["messages"][0]["content"], "原始完整系统提示。")

    def test_tool_error_can_complete_turn(self):
        self.script = [reply(None, [tool_call("1/0")]), reply("不能除以零。")]
        status, output = self.run_cli(["计算1/0", "exit"])
        turn = self.turns(self.read(self.ids(output)[0]))[0]
        self.assertEqual(turn["status"], "completed")
        self.assertIsNone(turn["runs"][0]["error_type"])
        self.assertFalse(json.loads(turn["messages"][2]["content"])["ok"])

    def test_running_record_exists_before_model_request(self):
        seen = []
        real_respond = self.respond

        def respond(request, timeout):
            snapshots = [json.loads(p.read_text(encoding="utf-8")) for p in self.session_root.glob("*.json")]
            seen.extend(t["status"] for s in snapshots for t in self.turns(s))
            return real_respond(request, timeout)

        self.transport.side_effect = respond
        self.script = [reply()]
        self.run_cli(["问题", "exit"])
        self.assertEqual(seen, ["running"])


def metered(response=None, prompt=10, completion=2):
    response = copy.deepcopy(reply() if response is None else response)
    response["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion,
                         "total_tokens": prompt + completion}
    return response


def as_legacy(session, version):
    """临时目录中的旧格式样例，不接触实际 Session。"""
    old = copy.deepcopy(session)
    old["version"] = version
    for thread in old["threads"].values():
        thread.pop("memory")
        for turn in thread["turns"]:
            turn.pop("kind")
            turn.pop("memory_update")
            run = turn["runs"][0]
            run.pop("task_type")
            if version == 1:
                for field in ("trace_available", "steps", "events", "usage"):
                    run.pop(field)
    return old


def expected_v4(old):
    result = copy.deepcopy(old)
    result["version"] = 4
    for thread in result["threads"].values():
        thread["memory"] = harness.empty_memory()
        for turn in thread["turns"]:
            turn.update(kind="chat", memory_update=None)
            turn["runs"][0]["task_type"] = "chat"
    return result


class TraceTests(OfflineCase):
    ids = SessionTests.ids
    read = SessionTests.read
    turns = SessionTests.turns
    completed_session = SessionTests.completed_session

    def setUp(self):
        super().setUp()
        self.store = harness.Store(self.session_root)

    def saved_run(self, question="问题"):
        status, output = self.run_cli([question, "exit"])
        session = self.read(self.ids(output)[0])
        return status, output, session, self.turns(session)[-1]["runs"][0]

    def legacy(self, running=False):
        session = self.completed_session()
        if running:
            harness.begin_turn(session, self.store, "旧未完成任务")
        session = as_legacy(session, 1)
        self.store.save(session)
        return session, self.store.path(session["id"]).read_bytes()

    def test_direct_answer_trace_and_cache_fields_not_double_counted(self):
        response = metered(prompt=10, completion=2)
        response["usage"].update(prompt_cache_hit_tokens=8, prompt_cache_miss_tokens=2)
        self.script = [response]
        status, output, session, run = self.saved_run()
        self.assertEqual(status, 0)
        self.assertEqual(session["version"], 4)
        self.assertEqual([s["step_type"] for s in run["steps"]], ["llm"])
        self.assertEqual(run["usage"], {"prompt_tokens": 10, "completion_tokens": 2,
                         "total_tokens": 12, "model_calls": 1, "missing_usage_calls": 0, "complete": True})
        self.assertIn("完整统计", output)
        self.assertEqual([e["event_type"] for e in run["events"]],
                         ["run.started", "step.started", "llm.usage", "step.completed", "run.completed"])
        for index, event in enumerate(run["events"], 1):
            self.assertEqual(event["sequence"], index)
            self.assertEqual(event["session_id"], session["id"])
            self.assertEqual(event["thread_id"], session["active_thread"])
            self.assertEqual(event["turn_id"], self.turns(session)[0]["id"])
            self.assertEqual(event["run_id"], run["id"])
            self.assertIsNotNone(datetime.fromisoformat(event["timestamp"]).utcoffset())
        raw = self.store.path(session["id"]).read_text(encoding="utf-8")
        for forbidden in (FAKE_KEY, "Authorization", "Bearer ", "DEEPSEEK_API_KEY", "prompt_cache_hit_tokens"):
            self.assertNotIn(forbidden, raw)

    def test_calculation_three_steps_and_all_model_usage(self):
        self.script = [metered(reply(None, [tool_call()]), 10, 1), metered(reply("1081"), 20, 3)]
        status, output, session, run = self.saved_run("计算23×47")
        self.assertEqual(status, 0)
        self.assertEqual([s["step_type"] for s in run["steps"]], ["llm", "tool", "llm"])
        self.assertEqual([s["status"] for s in run["steps"]], ["completed"] * 3)
        self.assertEqual(run["usage"]["total_tokens"], 34)
        self.assertEqual(run["usage"]["prompt_tokens"], 30)
        self.assertEqual(run["usage"]["completion_tokens"], 4)
        self.assertEqual(run["usage"]["model_calls"], 2)
        self.assertEqual(run["steps"][1]["input_data"]["tool_call_id"], "call_calc_1")
        self.assertEqual(run["steps"][1]["output_data"]["result"]["value"], 1081)
        self.assertEqual(len(run["steps"][0]["input_data"]["messages"]), 2)
        self.assertEqual(len(run["steps"][2]["input_data"]["messages"]), 4)
        self.assertEqual(run["steps"][2]["input_data"], self.requests[1])
        self.assertEqual(len({s["id"] for s in run["steps"]}), 3)
        self.assertEqual(len({e["id"] for e in run["events"]}), len(run["events"]))
        harness.validate_session(session)

    def test_three_tools_weather_is_one_step_and_ids_link(self):
        calls = [time_call(), tool_call(), weather_call()]
        self.script = [metered(reply(None, calls)), metered(reply("三个结果。"))]
        self.weather_script = [GEOCODING_DATA, WEATHER_DATA]
        status, output, session, run = self.saved_run("三个工具")
        self.assertEqual(status, 0)
        self.assertEqual([s["step_type"] for s in run["steps"]], ["llm", "tool", "tool", "tool", "llm"])
        self.assertEqual([s["input_data"]["tool_call_id"] for s in run["steps"][1:4]],
                         [call["id"] for call in calls])
        self.assertEqual(len(self.weather_requests), 2)
        self.assertEqual(run["usage"]["model_calls"], 2)
        self.assertEqual(run["usage"]["total_tokens"], 24)

    def test_step_is_durable_before_model_and_tool_execution(self):
        seen = []
        real_respond, real_execute = self.respond, main.execute

        def snapshot(kind):
            sessions = [json.loads(p.read_text(encoding="utf-8")) for p in self.session_root.glob("ses_*.json")]
            step = self.turns(sessions[0])[0]["runs"][0]["steps"][-1]
            seen.append((step["step_type"], step["status"]))
            self.assertEqual(step["step_type"], kind)
            self.assertEqual(step["status"], "running")

        def respond(request, timeout):
            snapshot("llm")
            return real_respond(request, timeout)

        def execute(name, arguments):
            snapshot("tool")
            return real_execute(name, arguments)

        self.transport.side_effect = respond
        self.script = [reply(None, [tool_call()]), reply("1081")]
        with patch.object(main, "execute", side_effect=execute):
            self.saved_run()
        self.assertEqual(seen, [("llm", "running"), ("tool", "running"), ("llm", "running")])

    def test_missing_usage_remains_unknown(self):
        self.script = [reply()]
        status, output, session, run = self.saved_run()
        self.assertEqual(run["usage"]["model_calls"], 1)
        self.assertEqual(run["usage"]["missing_usage_calls"], 1)
        self.assertFalse(run["usage"]["complete"])
        self.assertTrue(all(run["usage"][field] is None for field in harness.TOKEN_FIELDS))
        self.assertEqual(run["steps"][0]["usage"]["state"], "missing")
        self.assertIn("总计 未知", output)

    def test_partial_and_missing_usage_show_known_subtotals(self):
        response = reply(None, [tool_call()])
        response["usage"] = {"prompt_tokens": 8}
        self.script = [response, reply("1081")]
        status, output, session, run = self.saved_run()
        self.assertEqual(run["usage"]["prompt_tokens"], 8)
        self.assertIsNone(run["usage"]["completion_tokens"])
        self.assertIsNone(run["usage"]["total_tokens"])
        self.assertEqual(run["usage"]["missing_usage_calls"], 2)
        self.assertIn("数字仅为已知小计", output)

    def test_zero_is_valid_reported_usage(self):
        self.script = [metered(prompt=0, completion=0)]
        status, output, session, run = self.saved_run()
        self.assertIs(run["usage"]["complete"], True)
        self.assertEqual(run["usage"]["total_tokens"], 0)

    def test_invalid_usage_is_not_added_or_saved_raw(self):
        invalid = [[], "bad-usage", {"prompt_tokens": True}, {"prompt_tokens": -1},
                   {"prompt_tokens": "10"}, {"total_tokens": 2.5},
                   {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 999}]
        for usage in invalid:
            with self.subTest(usage=usage):
                response = reply()
                response["usage"] = usage
                self.script = [response]
                status, output, session, run = self.saved_run()
                self.assertEqual(status, 0)
                self.assertEqual(run["steps"][0]["usage"]["state"], "invalid")
                self.assertIsNone(run["usage"]["total_tokens"])
                self.assertFalse(run["usage"]["complete"])

    def test_truncated_or_invalid_response_keeps_parsed_usage(self):
        for response in (metered(reply("未完成", reason="length")),
                         {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}):
            with self.subTest(response=response):
                self.script = [response]
                status, output, session, run = self.saved_run()
                self.assertEqual(run["status"], "failed")
                self.assertEqual(run["steps"][0]["status"], "failed")
                self.assertEqual(run["steps"][0]["error_type"], "ResponseError")
                self.assertEqual(run["usage"]["total_tokens"], 12)
                self.assertTrue(run["usage"]["complete"])
                self.assertEqual(self.turns(session)[0]["messages"], [])
                self.assertIsNone(run["steps"][0]["output_data"])

    def test_http_and_unparsed_responses_have_unknown_usage_and_safe_errors(self):
        for error in (HTTPError(main.API_URL, 401, FAKE_KEY, {"Authorization": FAKE_KEY}, None),
                      TimeoutError(FAKE_KEY), URLError(FAKE_KEY), b"{bad", b"\xff"):
            with self.subTest(error=type(error).__name__):
                self.script = [error]
                status, output, session, run = self.saved_run()
                self.assertEqual(run["status"], "failed")
                self.assertEqual(run["usage"]["model_calls"], 1)
                self.assertIsNone(run["usage"]["total_tokens"])
                self.assertEqual(run["steps"][0]["usage"]["state"], "unreported")
                self.assertNotIn(FAKE_KEY, json.dumps(session))

    def test_failure_after_tool_preserves_steps_usage_and_excludes_history(self):
        self.script = [metered(reply(None, [tool_call()])), TimeoutError(FAKE_KEY), metered(reply("下一轮"))]
        status, output = self.run_cli(["失败的问题", "新的问题", "exit"])
        session = self.read(self.ids(output)[0])
        run = self.turns(session)[0]["runs"][0]
        self.assertEqual([s["status"] for s in run["steps"]], ["completed", "completed", "failed"])
        self.assertEqual(run["steps"][1]["output_data"]["result"]["value"], 1081)
        self.assertEqual(run["usage"]["total_tokens"], 12)
        self.assertFalse(run["usage"]["complete"])
        self.assertEqual(run["usage"]["model_calls"], 2)
        self.assertEqual(self.turns(session)[0]["messages"], [])
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]], ["新的问题"])

    def test_invalid_tool_batch_records_failed_llm_and_executes_nothing(self):
        self.script = [metered(reply(None, [tool_call(), tool_call()]))]
        with patch.object(main, "execute") as execute:
            status, output, session, run = self.saved_run()
        execute.assert_not_called()
        self.assertEqual(len(run["steps"]), 1)
        self.assertEqual(run["steps"][0]["status"], "failed")
        self.assertEqual(run["usage"]["total_tokens"], 12)

    def test_tool_error_is_failed_step_but_explanation_completes_run(self):
        self.script = [metered(reply(None, [tool_call("1/0")])), metered(reply("不能除以零。"))]
        status, output, session, run = self.saved_run()
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["steps"][1]["status"], "failed")
        self.assertEqual(run["steps"][1]["error_type"], "ToolError")
        self.assertFalse(run["steps"][1]["output_data"]["ok"])
        self.assertIn("step.failed", [e["event_type"] for e in run["events"]])

    def test_limit_keeps_all_six_model_usages_without_sixth_tool(self):
        self.script = [metered(reply(None, [tool_call(call_id=f"call_{i}")]), 10, 1) for i in range(6)]
        status, output, session, run = self.saved_run()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["usage"]["model_calls"], 6)
        self.assertEqual(run["usage"]["total_tokens"], 66)
        self.assertEqual(sum(s["step_type"] == "tool" for s in run["steps"]), 5)
        self.assertTrue(all(s["status"] == "completed" for s in run["steps"]))

    def test_cancel_preserves_prior_tool_and_usage(self):
        self.script = [metered(reply(None, [tool_call()])), KeyboardInterrupt()]
        status, output = self.run_cli(["取消的问题"])
        self.assertEqual(status, 130)
        session = self.read(self.ids(output)[0])
        run = self.turns(session)[0]["runs"][0]
        self.assertEqual(run["status"], "interrupted")
        self.assertEqual([s["status"] for s in run["steps"]], ["completed", "completed", "interrupted"])
        self.assertEqual(run["usage"]["total_tokens"], 12)
        self.assertFalse(run["usage"]["complete"])
        self.assertEqual(run["events"][-1]["event_type"], "run.interrupted")

    def test_recover_running_step_keeps_usage_and_does_not_reexecute(self):
        session = self.store.create("原提示")
        turn = harness.begin_turn(session, self.store, "未完成", traced=True)
        recorder = harness.Recorder(session, self.store, turn)
        step = recorder.start("llm", main.model_payload(harness.context(session), "offline-model"))
        recorder.usage(step, {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        loaded = self.store.load(session["id"])
        run = self.turns(loaded)[0]["runs"][0]
        self.assertEqual(run["steps"][0]["status"], "interrupted")
        self.assertEqual(run["steps"][0]["error_type"], "Interrupted")
        self.assertEqual(run["usage"]["total_tokens"], 12)
        self.assertEqual(run["events"][-2]["event_type"], "step.interrupted")
        self.assertEqual(run["events"][-1]["event_type"], "run.interrupted")
        self.assertEqual(len(harness.context(loaded)), 1)
        self.transport.assert_not_called()

    def test_runs_trace_help_are_local_and_create_no_turns(self):
        status, output = self.run_cli(["/runs", "/trace", "/trace run_000000000000", "help", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("当前 Thread 尚无 Run", output)
        self.assertIn("/trace RunID", output)
        self.assertEqual(self.turns(self.read(self.ids(output)[0])), [])
        self.transport.assert_not_called()

    def test_trace_latest_and_selected_after_reload(self):
        self.script = [metered(reply("回答一")), metered(reply("回答二"))]
        status, output = self.run_cli(["问题一", "问题二", "exit"])
        sid = self.ids(output)[0]
        session = self.read(sid)
        first, last = harness.current_runs(session)
        self.requests.clear()
        status, output = self.run_cli(["/load " + sid, "/runs", "/trace", "/trace " + first["id"],
                                       "/trace run_000000000000", "exit"])
        self.assertEqual(status, 0)
        self.assertIn(first["id"], output)
        self.assertIn(last["id"], output)
        self.assertIn("未找到此 Run ID", output)
        self.assertIn("回答一", harness.trace_text(session, first["id"]))
        self.assertIn("回答二", harness.trace_text(session))
        self.assertEqual(len(self.turns(self.read(sid))), 2)
        self.assertEqual(self.requests, [])

    def test_trace_data_does_not_enter_next_model_context(self):
        self.script = [metered(reply("回答一")), metered(reply("回答二"))]
        status, output = self.run_cli(["问题一", "/trace", "/runs", "问题二", "exit"])
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]],
                         ["问题一", "回答一", "问题二"])
        self.assertNotIn("step_", json.dumps(self.requests[-1]["messages"]))

    def test_v1_backup_preserves_bytes_ids_prompt_and_context(self):
        old, original = self.legacy()
        loaded = self.store.load(old["id"])
        self.assertEqual(loaded["version"], 4)
        for field in ("id", "active_thread", "system_prompt", "created_at"):
            self.assertEqual(loaded[field], old[field])
        self.assertEqual(harness.context(loaded), harness.context(old))
        self.assertEqual(self.turns(loaded)[0]["id"], self.turns(old)[0]["id"])
        run = self.turns(loaded)[0]["runs"][0]
        self.assertEqual(run["id"], self.turns(old)[0]["runs"][0]["id"])
        self.assertFalse(run["trace_available"])
        self.assertEqual(run["steps"], [])
        self.assertEqual(run["events"], [])
        self.assertIsNone(run["usage"]["model_calls"])
        self.assertIn("不补造旧执行记录", harness.trace_text(loaded))
        backup = self.store.path(old["id"]).with_suffix(".v1.bak.json")
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(self.store.load(old["id"]), loaded)
        self.assertEqual(backup.read_bytes(), original)
        self.transport.assert_not_called()

    def test_migrated_v1_accepts_new_traced_run_without_fabricating_old_usage(self):
        old, original = self.legacy()
        self.script = [metered(reply("新回答"))]
        status, output = self.run_cli(["/load " + old["id"], "新问题", "exit"])
        self.assertEqual(status, 0)
        loaded = self.read(old["id"])
        first, last = harness.current_runs(loaded)
        self.assertFalse(first["trace_available"])
        self.assertTrue(last["trace_available"])
        self.assertEqual(last["usage"]["total_tokens"], 12)
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]],
                         ["旧问题", "旧回答", "新问题"])
        self.assertEqual(self.store.path(old["id"]).with_suffix(".v1.bak.json").read_bytes(), original)

    def test_v1_running_is_interrupted_after_backup_without_replay(self):
        old, original = self.legacy(running=True)
        loaded = self.store.load(old["id"])
        run = self.turns(loaded)[-1]["runs"][0]
        self.assertEqual(run["status"], "interrupted")
        self.assertFalse(run["trace_available"])
        self.assertEqual(run["events"], [])
        self.assertEqual(self.store.path(old["id"]).with_suffix(".v1.bak.json").read_bytes(), original)
        self.transport.assert_not_called()
        self.weather_transport.assert_not_called()

    def test_invalid_v1_is_rejected_before_backup(self):
        old, original = self.legacy()
        self.turns(old)[0]["messages"] = []
        raw = json.dumps(old).encode("utf-8")
        target = self.store.path(old["id"])
        target.write_bytes(raw)
        with self.assertRaises(harness.SessionError):
            self.store.load(old["id"])
        self.assertEqual(target.read_bytes(), raw)
        self.assertFalse(target.with_suffix(".v1.bak.json").exists())

    def test_v1_backup_failure_preserves_original(self):
        old, original = self.legacy()
        with patch.object(harness.os, "rename", side_effect=OSError(FAKE_KEY)):
            with self.assertRaises(harness.StorageError) as caught:
                self.store.load(old["id"])
        self.assertNotIn(FAKE_KEY, str(caught.exception))
        self.assertEqual(self.store.path(old["id"]).read_bytes(), original)
        self.assertEqual(list(self.session_root.glob("*.tmp")), [])

    def test_v1_upgrade_write_failure_preserves_original_and_completed_backup(self):
        old, original = self.legacy()
        with patch.object(harness.os, "replace", side_effect=OSError(FAKE_KEY)):
            with self.assertRaises(harness.StorageError):
                self.store.load(old["id"])
        target = self.store.path(old["id"])
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(target.with_suffix(".v1.bak.json").read_bytes(), original)
        self.assertEqual(list(self.session_root.glob("*.tmp")), [])
        self.assertEqual(self.store.load(old["id"])["version"], 4)

    def test_different_existing_v1_backup_is_never_overwritten(self):
        old, original = self.legacy()
        target = self.store.path(old["id"])
        backup = target.with_suffix(".v1.bak.json")
        backup.write_bytes(b"earlier-backup")
        with self.assertRaises(harness.StorageError):
            self.store.load(old["id"])
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), b"earlier-backup")

    def test_cli_migration_failure_does_not_claim_loaded(self):
        old, original = self.legacy()
        with patch.object(harness.Store, "backup_legacy", side_effect=harness.StorageError("备份失败，停止。")):
            status, output = self.run_cli(["/load " + old["id"], "不应继续"])
        self.assertEqual(status, 1)
        self.assertNotIn("会话已恢复", output)
        self.assertEqual(self.store.path(old["id"]).read_bytes(), original)
        self.transport.assert_not_called()

    def test_corrupt_steps_events_or_usage_are_rejected_without_overwrite(self):
        self.script = [metered(reply(None, [tool_call()])), metered(reply("1081"))]
        status, output, session, run = self.saved_run()
        target = self.store.path(session["id"])
        changes = [lambda r: r.update(usage={}), lambda r: r.update(steps=[{}]),
                   lambda r: r["usage"].update(total_tokens=999),
                   lambda r: r["steps"][0]["input_data"].update(Authorization=FAKE_KEY),
                   lambda r: r["steps"][0].update(status=[]),
                   lambda r: r["steps"][0]["usage"].update(prompt_tokens=True),
                   lambda r: r["events"][1].update(sequence=1),
                   lambda r: r["events"][1].update(step_id="step_000000000000"),
                   lambda r: r["events"][1].update(thread_id="thr_000000000000"),
                   lambda r: r["events"][2]["data"].update(total_tokens=999),
                   lambda r: r["events"].pop(),
                   lambda r: r["steps"][1].update(id=r["steps"][0]["id"]),
                   lambda r: r.update(trace_available=False)]
        for change in changes:
            bad = copy.deepcopy(session)
            change(self.turns(bad)[0]["runs"][0])
            raw = json.dumps(bad).encode("utf-8")
            target.write_bytes(raw)
            with self.subTest(change=changes.index(change)), self.assertRaises(harness.SessionError):
                self.store.load(session["id"])
            self.assertEqual(target.read_bytes(), raw)

    def test_tool_record_save_failure_stops_before_followup_request(self):
        real_save = harness.Store.save
        # 第6次写盘为 tool.started；第7次为 tool.completed。
        for failing_call, expected_executions in ((6, 0), (7, 1)):
            calls = []

            def save(store, session):
                calls.append(session["id"])
                if len(calls) == failing_call:
                    raise harness.StorageError("模拟工具记录保存失败，停止。")
                return real_save(store, session)

            self.requests.clear()
            self.script = [metered(reply(None, [tool_call()])), metered(reply("不应调用"))]
            with self.subTest(failing_call=failing_call), patch.object(harness.Store, "save", save), \
                    patch.object(main, "execute", wraps=main.execute) as execute:
                status, output = self.run_cli(["计算", "不应继续"])
            self.assertEqual(status, 1)
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(execute.call_count, expected_executions)
            self.assertEqual(len(calls), failing_call)
            self.assertNotIn("模型回答：", output)

    def test_failure_record_save_failure_stops_without_retrying_save(self):
        real_save = harness.Store.save
        calls = []

        def save(store, session):
            calls.append(session["id"])
            if len(calls) == 4:
                raise harness.StorageError("模拟失败记录保存失败，停止。")
            return real_save(store, session)

        self.script = [TimeoutError(FAKE_KEY)]
        with patch.object(harness.Store, "save", save):
            status, output = self.run_cli(["失败", "不应继续"])
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.turns(self.read(calls[0]))[0]["status"], "running")

    def test_v1_tool_and_failed_turns_keep_original_protocol(self):
        self.script = [reply(None, [tool_call()]), reply("1081"), TimeoutError(FAKE_KEY)]
        status, output = self.run_cli(["旧计算", "旧失败", "exit"])
        old = self.read(self.ids(output)[0])
        old = as_legacy(old, 1)
        self.store.save(old)
        original = self.store.path(old["id"]).read_bytes()
        loaded = self.store.load(old["id"])
        self.assertEqual(harness.context(loaded), harness.context(old))
        self.assertEqual([m["role"] for m in harness.context(loaded)],
                         ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(self.turns(loaded)[1]["status"], "failed")
        self.assertEqual(self.turns(loaded)[1]["messages"], [])
        self.assertEqual(self.store.path(old["id"]).with_suffix(".v1.bak.json").read_bytes(), original)


class ThreadTests(OfflineCase):
    ids = SessionTests.ids
    read = SessionTests.read
    turns = SessionTests.turns
    completed_session = SessionTests.completed_session

    def setUp(self):
        super().setUp()
        self.store = harness.Store(self.session_root)

    def scripted_inputs(self, items):
        iterator = iter(items)

        def take(prompt):
            item = next(iterator)
            return item() if callable(item) else item
        return take

    def old_v2(self, running=False):
        self.script = [metered(reply(None, [tool_call()])), metered(reply("旧计算1081")), TimeoutError(FAKE_KEY)]
        status, output = self.run_cli(["旧计算", "旧失败", "exit"])
        session = self.read(self.ids(output)[0])
        if running:
            turn = harness.begin_turn(session, self.store, "旧未完成", traced=True)
            recorder = harness.Recorder(session, self.store, turn)
            step = recorder.start("llm", main.model_payload(harness.context(session), "offline-model"))
            recorder.usage(step, {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})
        session = as_legacy(session, 2)
        self.store.save(session)
        return session, self.store.path(session["id"]).read_bytes()

    def test_new_thread_keeps_session_prompt_and_original_history(self):
        original = self.completed_session("共同系统提示")
        old = copy.deepcopy(original)
        a = original["active_thread"]
        candidate = harness.new_thread(original, self.store)
        b = candidate["active_thread"]
        self.assertNotEqual(a, b)
        self.assertEqual(original, old)
        self.assertEqual(candidate["id"], original["id"])
        self.assertEqual(candidate["system_prompt"], "共同系统提示")
        self.assertEqual(candidate["threads"][a], old["threads"][a])
        self.assertEqual(candidate["threads"][b]["turns"], [])
        self.assertEqual(harness.context(candidate), [{"role": "system", "content": "共同系统提示"}])
        self.assertEqual(self.read(candidate["id"]), candidate)

    def test_a_and_b_preferences_are_isolated_in_actual_request_messages(self):
        initial = self.store.create("共同系统提示")
        sid, a = initial["id"], initial["active_thread"]
        self.script = [metered(reply("已记住37")), metered(reply("本Thread尚不知道")),
                       metered(reply("已记住92")), metered(reply("你喜欢37")), metered(reply("你喜欢92"))]
        get_b = lambda: next(tid for tid in self.read(sid)["threads"] if tid != a)
        status, output = self.run_cli(self.scripted_inputs([
            "/load " + sid, "我喜欢37", "/thread", "本Thread我喜欢什么", "我喜欢92",
            "/use " + a, "再问A", lambda: "/use " + get_b(), "再问B", "exit",
        ]))
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 5)
        self.assertEqual(self.requests[1]["messages"], [
            {"role": "system", "content": "共同系统提示"}, {"role": "user", "content": "本Thread我喜欢什么"}])
        self.assertEqual([m["content"] for m in self.requests[3]["messages"][1:]],
                         ["我喜欢37", "已记住37", "再问A"])
        self.assertEqual([m["content"] for m in self.requests[4]["messages"][1:]],
                         ["本Thread我喜欢什么", "本Thread尚不知道", "我喜欢92", "已记住92", "再问B"])
        loaded = self.read(sid)
        self.assertEqual(len(loaded["threads"][a]["turns"]), 2)
        self.assertEqual(len(loaded["threads"][get_b()]["turns"]), 3)
        self.assertEqual(loaded["active_thread"], get_b())
        self.assertTrue(all(request["messages"][0]["content"] == "共同系统提示" for request in self.requests))
        self.assertNotIn(FAKE_KEY, json.dumps(loaded))

    def test_new_thread_does_not_copy_tool_messages_or_runs(self):
        self.script = [metered(reply(None, [tool_call()])), metered(reply("1081")), metered(reply("新回答"))]
        status, output = self.run_cli(["计算23×47", "/thread", "新问题", "exit"])
        sid = self.ids(output)[0]
        saved = self.read(sid)
        a, b = saved["threads"]
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]], ["新问题"])
        self.assertEqual(len(saved["threads"][a]["turns"][0]["runs"][0]["steps"]), 3)
        self.assertEqual(len(saved["threads"][b]["turns"]), 1)
        for tid, thread in saved["threads"].items():
            for turn in thread["turns"]:
                for event in turn["runs"][0]["events"]:
                    self.assertEqual(event["thread_id"], tid)

    def test_local_thread_commands_create_no_turns_runs_or_model_requests(self):
        status, output = self.run_cli(["help", "/threads", "/thread", "/threads", "/session", "/runs", "/trace", "exit"])
        self.assertEqual(status, 0)
        saved = self.read(self.ids(output)[0])
        self.assertEqual(len(saved["threads"]), 2)
        self.assertTrue(all(thread["turns"] == [] for thread in saved["threads"].values()))
        self.assertIn("* 表示当前选中", output)
        self.assertIn("/use ThreadID", output)
        self.assertIn("当前 Thread 尚无 Run", output)
        self.assertEqual(output.count("已创建并切换到空 Thread"), 1)
        self.transport.assert_not_called()
        self.weather_transport.assert_not_called()

    def test_thread_list_marks_only_current_and_reports_completed_and_failed_counts(self):
        session = self.completed_session()
        a = session["active_thread"]
        failed = harness.begin_turn(session, self.store, "失败")
        harness.finish_turn(session, self.store, failed, "failed", error_type="TimeoutError")
        session = harness.new_thread(session, self.store)
        b = session["active_thread"]
        output = harness.threads_text(session)
        self.assertIn(f"* {b}；已完成问答 0 轮；记录轮次 0 轮", output)
        self.assertIn(f"  {a}；已完成问答 1 轮；记录轮次 2 轮", output)
        self.assertEqual(output.count("  * "), 1)

    def test_invalid_or_foreign_thread_ids_leave_selection_and_file_unchanged(self):
        session = self.completed_session()
        foreign = self.store.create("另一个Session")
        target = self.store.path(session["id"])
        original = target.read_bytes()
        snapshot = copy.deepcopy(session)
        for tid in ("../.env", "thr_ABCDEF123456", "thr_short", "thr_000000000000",
                    session["id"], foreign["active_thread"], None, []):
            with self.subTest(tid=tid), self.assertRaises(harness.SessionError):
                harness.use_thread(session, self.store, tid)
            self.assertEqual(session, snapshot)
            self.assertEqual(target.read_bytes(), original)
        self.transport.assert_not_called()

    def test_cli_bad_thread_arguments_are_local_and_keep_context(self):
        initial = self.completed_session()
        self.script = [reply("仍然使用原历史")]
        status, output = self.run_cli(["/load " + initial["id"], "/thread extra", "/threads extra",
                                       "/use", "/use a b", "/use invalid", "/use thr_000000000000",
                                       "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("命令失败：", output)
        saved = self.read(initial["id"])
        self.assertEqual(saved["active_thread"], initial["active_thread"])
        self.assertEqual(len(saved["threads"]), 1)
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]], ["旧问题", "旧回答", "新问题"])

    def test_switch_to_same_thread_is_noop(self):
        session = self.completed_session()
        with patch.object(self.store, "save") as save:
            result = harness.use_thread(session, self.store, session["active_thread"])
        self.assertIs(result, session)
        save.assert_not_called()

    def test_create_and_switch_save_failures_preserve_memory_and_disk(self):
        session = self.completed_session()
        a = session["active_thread"]
        session = harness.new_thread(session, self.store)
        target = self.store.path(session["id"])
        for action in (lambda: harness.new_thread(session, self.store),
                       lambda: harness.use_thread(session, self.store, a)):
            for failure in ("replace", "fsync"):
                before = copy.deepcopy(session)
                original = target.read_bytes()
                with self.subTest(failure=failure), patch.object(harness.os, failure, side_effect=OSError(FAKE_KEY)):
                    with self.assertRaises(harness.StorageError) as caught:
                        action()
                self.assertNotIn(FAKE_KEY, str(caught.exception))
                self.assertEqual(session, before)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(list(self.session_root.glob("*.tmp")), [])

    def test_cli_stops_without_claiming_create_or_switch_success_on_save_failure(self):
        original = self.completed_session()
        a = original["active_thread"]
        original = harness.new_thread(original, self.store)
        sid = original["id"]
        real_save = harness.Store.save

        def save(store, candidate):
            if candidate["id"] == sid:
                raise harness.StorageError("模拟Thread保存失败，停止。")
            return real_save(store, candidate)

        for command in ("/thread", "/use " + a):
            before = self.store.path(sid).read_bytes()
            with self.subTest(command=command), patch.object(harness.Store, "save", save):
                status, output = self.run_cli(["/load " + sid, command, "不应继续"])
            self.assertEqual(status, 1)
            self.assertNotIn("已创建并切换", output)
            self.assertNotIn("当前 Thread 已选中", output)
            self.assertEqual(self.store.path(sid).read_bytes(), before)
        self.transport.assert_not_called()

    def test_runs_and_trace_follow_selected_thread(self):
        initial = self.store.create("共同提示")
        sid, a = initial["id"], initial["active_thread"]
        self.script = [metered(reply("A回答")), metered(reply("B回答"))]
        self.run_cli(["/load " + sid, "A问题", "/thread", "B问题", "exit"])
        session = self.read(sid)
        b = session["active_thread"]
        a_run = session["threads"][a]["turns"][0]["runs"][0]
        b_run = session["threads"][b]["turns"][0]["runs"][0]
        self.assertEqual(harness.current_runs(session), [b_run])
        self.assertIn("B回答", harness.trace_text(session))
        self.assertNotIn("A回答", harness.trace_text(session))
        self.assertIn("未找到此 Run ID", harness.trace_text(session, a_run["id"]))
        session = harness.use_thread(session, self.store, a)
        self.assertEqual(harness.current_runs(session), [a_run])
        self.assertIn("A回答", harness.trace_text(session, a_run["id"]))
        self.assertIn("未找到此 Run ID", harness.trace_text(session, b_run["id"]))
        self.requests.clear()
        status, output = self.run_cli(["/load " + sid, "/runs", "/trace " + b_run["id"], "/use " + b,
                                       "/runs", "/trace", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("未找到此 Run ID", output)
        self.assertIn(a_run["id"], output)
        self.assertIn(b_run["id"], output)
        self.assertEqual(self.requests, [])
        self.assertEqual([len(t["turns"]) for t in self.read(sid)["threads"].values()], [1, 1])

    def test_restart_restores_b_selection_then_a_history_and_tool_protocol(self):
        initial = self.store.create("原完整提示")
        sid, a = initial["id"], initial["active_thread"]
        self.script = [metered(reply(None, [tool_call()])), metered(reply("A是1081")), metered(reply("B是92"))]
        self.run_cli(["/load " + sid, "A计算", "/thread", "B偏好92", "exit"])
        b = self.read(sid)["active_thread"]
        self.script = [metered(reply("B恢复")), metered(reply("A恢复1081"))]
        status, output = self.run_cli(["/load " + sid, "继续B", "/use " + a, "继续A", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual([m["content"] for m in self.requests[-2]["messages"][1:]], ["B偏好92", "B是92", "继续B"])
        a_messages = self.requests[-1]["messages"]
        self.assertEqual([m["role"] for m in a_messages], ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(a_messages[3]["tool_call_id"], "call_calc_1")
        self.assertEqual(a_messages[4]["content"], "A是1081")
        self.assertEqual(a_messages[0]["content"], "原完整提示")
        loaded = self.store.load(sid)
        self.assertEqual(loaded["active_thread"], a)
        self.assertEqual(len(loaded["threads"][b]["turns"]), 2)

    def test_new_session_from_multithread_keeps_old_file_and_has_one_empty_thread(self):
        original = harness.new_thread(self.completed_session(), self.store)
        sid = original["id"]
        before = self.store.path(sid).read_bytes()
        status, output = self.run_cli(["/load " + sid, "/new", "/threads", "exit"])
        self.assertEqual(status, 0)
        new_id = self.ids(output)[-1]
        self.assertNotEqual(new_id, sid)
        new = self.read(new_id)
        self.assertEqual(len(new["threads"]), 1)
        self.assertEqual(self.turns(new), [])
        self.assertEqual(self.store.path(sid).read_bytes(), before)
        self.transport.assert_not_called()

    def test_failed_turn_is_retained_only_in_its_thread_and_excluded_from_context(self):
        initial = self.store.create("共同提示")
        sid, a = initial["id"], initial["active_thread"]
        self.script = [metered(reply("A回答")), TimeoutError(FAKE_KEY), metered(reply("B回答")), metered(reply("A继续"))]
        status, output = self.run_cli(["/load " + sid, "A问题", "/thread", "B失败", "B新问题",
                                       "/use " + a, "A新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual([m["content"] for m in self.requests[-2]["messages"][1:]], ["B新问题"])
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][1:]], ["A问题", "A回答", "A新问题"])
        saved = self.read(sid)
        b = next(tid for tid in saved["threads"] if tid != a)
        self.assertEqual([t["status"] for t in saved["threads"][b]["turns"]], ["failed", "completed"])
        self.assertEqual(saved["threads"][b]["turns"][0]["messages"], [])

    def test_load_recovers_running_records_in_all_threads_with_correct_event_owners(self):
        session = self.completed_session("共同提示")
        a = session["active_thread"]
        turn_a = harness.begin_turn(session, self.store, "A未完成", traced=True)
        recorder = harness.Recorder(session, self.store, turn_a)
        step = recorder.start("llm", main.model_payload(harness.context(session), "offline-model"))
        recorder.usage(step, {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8})
        session = harness.new_thread(session, self.store)
        b = session["active_thread"]
        turn_b = harness.begin_turn(session, self.store, "B未完成工具", traced=True)
        harness.Recorder(session, self.store, turn_b).start("tool", {
            "tool_call_id": "call_unfinished", "name": "get_time", "arguments": "{}"})
        loaded = self.store.load(session["id"])
        self.assertEqual(loaded["active_thread"], b)
        for tid in (a, b):
            run = loaded["threads"][tid]["turns"][-1]["runs"][0]
            self.assertEqual(run["status"], "interrupted")
            self.assertEqual(run["steps"][-1]["status"], "interrupted")
            self.assertEqual(run["events"][-1]["event_type"], "run.interrupted")
            self.assertTrue(all(e["thread_id"] == tid for e in run["events"]))
        self.assertEqual(loaded["threads"][a]["turns"][-1]["runs"][0]["usage"]["total_tokens"], 8)
        self.assertEqual(len(harness.context(loaded)), 1)
        self.assertEqual(len(harness.context(harness.use_thread(loaded, self.store, a))), 3)
        self.transport.assert_not_called()
        self.weather_transport.assert_not_called()
        self.clock.assert_not_called()
        harness.validate_session(loaded)

    def test_multithread_recovery_save_failure_preserves_snapshot(self):
        session = self.store.create("共同提示")
        turn = harness.begin_turn(session, self.store, "非当前Thread未完成", traced=True)
        harness.Recorder(session, self.store, turn).start("llm", main.model_payload(harness.context(session), "offline-model"))
        session = harness.new_thread(session, self.store)
        before = self.store.path(session["id"]).read_bytes()
        with patch.object(harness.os, "replace", side_effect=OSError(FAKE_KEY)):
            with self.assertRaises(harness.StorageError):
                self.store.load(session["id"])
        self.assertEqual(self.store.path(session["id"]).read_bytes(), before)
        self.assertEqual(self.store.load(session["id"])["active_thread"], session["active_thread"])
        self.transport.assert_not_called()

    def test_v2_upgrade_preserves_all_completed_and_failed_records_and_original_bytes(self):
        old, original = self.old_v2()
        loaded = self.store.load(old["id"])
        expected = expected_v4(old)
        self.assertEqual(loaded, expected)
        self.assertEqual(harness.context(loaded), harness.context(old))
        backup = self.store.path(old["id"]).with_suffix(".v2.bak.json")
        self.assertEqual(backup.read_bytes(), original)
        self.assertFalse(self.store.path(old["id"]).with_suffix(".v1.bak.json").exists())
        self.assertEqual(self.store.load(old["id"]), loaded)
        self.assertEqual(backup.read_bytes(), original)
        candidate = harness.new_thread(loaded, self.store)
        self.assertEqual(len(candidate["threads"]), 2)

    def test_v2_running_upgrade_keeps_usage_and_backs_up_before_recovery(self):
        old, original = self.old_v2(running=True)
        loaded = self.store.load(old["id"])
        run = self.turns(loaded)[-1]["runs"][0]
        self.assertEqual(run["status"], "interrupted")
        self.assertEqual(run["steps"][-1]["status"], "interrupted")
        self.assertEqual(run["usage"]["total_tokens"], 4)
        self.assertEqual(self.store.path(old["id"]).with_suffix(".v2.bak.json").read_bytes(), original)
        self.assertEqual(self.turns(loaded)[0], self.turns(expected_v4(old))[0])
        self.assertEqual(self.turns(loaded)[1], self.turns(expected_v4(old))[1])

    def test_v2_backup_and_upgrade_failures_keep_original_and_allow_retry(self):
        old, original = self.old_v2()
        target = self.store.path(old["id"])
        for failure in ("rename", "fsync", "replace"):
            with self.subTest(failure=failure), patch.object(harness.os, failure, side_effect=OSError(FAKE_KEY)):
                with self.assertRaises(harness.StorageError) as caught:
                    self.store.load(old["id"])
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(self.session_root.glob("*.tmp")), [])
        self.assertEqual(target.with_suffix(".v2.bak.json").read_bytes(), original)
        self.assertEqual(self.store.load(old["id"])["version"], 4)

    def test_different_v2_backup_is_never_overwritten_and_v1_backup_is_untouched(self):
        old, original = self.old_v2()
        target = self.store.path(old["id"])
        target.with_suffix(".v1.bak.json").write_bytes(b"original-phase-3.1-backup")
        backup = target.with_suffix(".v2.bak.json")
        backup.write_bytes(b"different-v2-backup")
        with self.assertRaises(harness.StorageError):
            self.store.load(old["id"])
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), b"different-v2-backup")
        self.assertEqual(target.with_suffix(".v1.bak.json").read_bytes(), b"original-phase-3.1-backup")

    def test_existing_v1_backup_survives_successful_v2_upgrade(self):
        old, original = self.old_v2()
        target = self.store.path(old["id"])
        first_backup = target.with_suffix(".v1.bak.json")
        first_backup.write_bytes(b"original-phase-3.1-backup")
        self.assertEqual(self.store.load(old["id"])["version"], 4)
        self.assertEqual(first_backup.read_bytes(), b"original-phase-3.1-backup")
        self.assertEqual(target.with_suffix(".v2.bak.json").read_bytes(), original)

    def test_invalid_v2_is_rejected_before_backup_or_overwrite(self):
        old, original = self.old_v2()
        self.turns(old)[0]["runs"][0]["usage"]["total_tokens"] = 999
        raw = json.dumps(old).encode("utf-8")
        target = self.store.path(old["id"])
        target.write_bytes(raw)
        with self.assertRaises(harness.SessionError):
            self.store.load(old["id"])
        self.assertEqual(target.read_bytes(), raw)
        self.assertFalse(target.with_suffix(".v2.bak.json").exists())

    def test_empty_or_invalid_threads_or_missing_selection_are_rejected(self):
        session = harness.new_thread(self.completed_session(), self.store)
        cases = []
        for field, value in (("threads", {}), ("threads", []), ("active_thread", "thr_000000000000"),
                             ("active_thread", [])):
            bad = copy.deepcopy(session)
            bad[field] = value
            cases.append(bad)
        for version in (1, 2):
            bad = copy.deepcopy(session)
            bad["version"] = version
            cases.append(bad)
        bad = copy.deepcopy(session)
        bad["threads"][bad["active_thread"]]["id"] = "thr_000000000000"
        cases.append(bad)
        for bad in cases:
            with self.subTest(case=cases.index(bad)), self.assertRaises(harness.SessionError):
                harness.validate_session(bad)

    def test_duplicate_record_ids_across_threads_are_rejected(self):
        initial = self.store.create("共同提示")
        self.script = [metered(reply("A回答")), metered(reply("B回答"))]
        self.run_cli(["/load " + initial["id"], "A问题", "/thread", "B问题", "exit"])
        valid = self.read(initial["id"])
        a, b = valid["threads"]
        for kind in ("turn", "run", "step", "event"):
            bad = copy.deepcopy(valid)
            ta, tb = bad["threads"][a]["turns"][0], bad["threads"][b]["turns"][0]
            ra, rb = ta["runs"][0], tb["runs"][0]
            if kind == "turn":
                tb["id"] = ta["id"]
                for event in rb["events"]:
                    event["turn_id"] = tb["id"]
            elif kind == "run":
                rb["id"] = ra["id"]
                for event in rb["events"]:
                    event["run_id"] = rb["id"]
            elif kind == "step":
                rb["steps"][0]["id"] = ra["steps"][0]["id"]
                for event in rb["events"]:
                    if event["step_id"] is not None:
                        event["step_id"] = rb["steps"][0]["id"]
            else:
                rb["events"][0]["id"] = ra["events"][0]["id"]
            with self.subTest(kind=kind), self.assertRaises(harness.SessionError):
                harness.validate_session(bad)


class MemoryTests(OfflineCase):
    ids = SessionTests.ids
    read = SessionTests.read
    turns = SessionTests.turns
    scripted_inputs = ThreadTests.scripted_inputs

    def setUp(self):
        super().setUp()
        self.store = harness.Store(self.session_root)

    def add_chat(self, session, text, answer="模拟回答", tool=False):
        def runner(messages):
            extra = [{"role": "user", "content": text}]
            if tool:
                extra += [{"role": "assistant", "content": None, "tool_calls": [tool_call()]},
                          {"role": "tool", "tool_call_id": "call_calc_1",
                           "content": json.dumps({"ok": True, "result": {"value": 1081}})}]
            return answer, messages + extra + [{"role": "assistant", "content": answer}]
        harness.run_saved_turn(session, self.store, text, runner)

    def seed(self, count=10, questions=None, tool_index=None):
        session = self.store.create("共同系统提示")
        for i in range(count):
            text = questions[i] if questions is not None else ("我喜欢的数字是37，请记住。" if i == 0 else f"问题{i+1}")
            self.add_chat(session, text, f"回答{i+1}", tool=i == tool_index)
        return session

    def fact(self, turn, value="37", key="favorite_number", index=0, kind="user"):
        return {"key": key, "value": value, "kind": kind, "sources": [{
            "turn_id": turn["id"], "message_index": index, "quote": turn["messages"][index]["content"]}]}

    def content(self, facts=None, summary="用户提供了数字偏好；讨论了若干问题。", removed=None):
        if summary and len(summary) <= harness.MAX_SUMMARY_CHARS and not summary.startswith("主题与目标："):
            summary = self.structured_summary(completed=summary)
        return {"facts": [] if facts is None else facts, "summary": summary,
                "removed_facts": [] if removed is None else removed}

    def structured_summary(self, topic="学习Python", completed="尚无已完成学习内容",
                           state="本批整理了对话", pending="无新的待办", suggestions="无", agreements=None):
        confirmed = "无" if not agreements else json.dumps(agreements, ensure_ascii=False)
        return (f"主题与目标：{topic}\n已完成内容：{completed}\n"
                f"当前状态：{state}\n待办与约定：\n待办：{pending}\n"
                f"助手建议（未确认）：{suggestions}\n用户明确约定：{confirmed}")

    def response(self, data):
        return metered(reply(json.dumps(data, ensure_ascii=False)), 100, 20)

    def compact(self, session, data):
        self.script = [self.response(data)]
        return harness.run_compaction(session, self.store, lambda plan, recorder:
            main.summarize_history(plan, recorder, FAKE_KEY, "offline-model"))[0]

    def test_compaction_keeps_originals_and_saves_sourced_memory_and_usage(self):
        session = self.seed()
        original = copy.deepcopy(self.turns(session))
        fact = self.fact(original[0])
        session = self.compact(session, self.content([fact]))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["revision"], 1)
        self.assertEqual(memory["covered_turn_ids"], [t["id"] for t in original[:5]])
        self.assertEqual(memory["facts"], [fact])
        self.assertEqual(self.turns(session)[:10], original)
        turn = self.turns(session)[-1]
        self.assertEqual(turn["kind"], "compaction")
        self.assertEqual(turn["messages"], [])
        self.assertEqual(turn["memory_update"], memory)
        run = turn["runs"][0]
        self.assertEqual(run["task_type"], "compaction")
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["usage"]["total_tokens"], 120)
        self.assertEqual(run["usage"]["model_calls"], 1)
        self.assertEqual(len(run["steps"]), 1)
        self.assertIn("记忆整理", harness.run_summary(run))
        self.assertEqual(self.store.load(session["id"]), session)
        messages = harness.context(session)
        self.assertEqual(len(messages), 12)
        self.assertIn("37", messages[1]["content"])
        self.assertEqual([m["content"] for m in messages[2::2]], [t["input_text"] for t in original[5:]])
        data = json.loads(self.requests[0]["messages"][1]["content"])
        self.assertEqual([item["turn_id"] for item in data["batch"]], memory["covered_turn_ids"])
        self.assertNotIn("source_turns", data)

    def test_small_history_and_view_commands_make_no_request_or_write(self):
        for count in (0, 5):
            session = self.seed(count)
            before = self.store.path(session["id"]).read_bytes()
            with patch.object(self.store, "save") as save:
                same, changed = harness.run_compaction(session, self.store, MagicMock())
            self.assertIs(same, session)
            self.assertFalse(changed)
            save.assert_not_called()
            status, output = self.run_cli(["/load " + session["id"], "/compact", "/memory", "/context", "help", "exit"])
            self.assertEqual(status, 0)
            self.assertIn("无需整理；未请求模型", output)
            self.assertIn("/compact会调用模型", output)
            self.assertIn("128 KiB", output)
            self.assertEqual(self.store.path(session["id"]).read_bytes(), before)
        self.transport.assert_not_called()

    def test_only_six_completed_turns_compacts_one_and_keeps_five(self):
        session = self.compact(self.seed(6), self.content())
        thread = session["threads"][session["active_thread"]]
        self.assertEqual(len(thread["memory"]["covered_turn_ids"]), 1)
        self.assertEqual(len(harness.context(session)), 12)

    def test_all_uncovered_turns_remain_after_new_chat_no_gap(self):
        session = self.compact(self.seed(), self.content())
        self.add_chat(session, "第11轮")
        messages = harness.context(session)
        self.assertEqual(len(messages), 14)
        self.assertEqual(messages[2]["content"], "问题6")
        self.assertEqual(messages[-2]["content"], "第11轮")
        self.assertIn("尚未压缩原文：6轮", harness.context_text(session))
        self.assertEqual(len(harness.completed_turns(session["threads"][session["active_thread"]])), 11)

    def test_batch_limit_and_rolling_summary_cover_contiguous_prefix(self):
        session = self.seed(16)
        session = self.compact(session, self.content(summary="摘要一"))
        session = self.compact(session, self.content(summary="摘要二：延续此前讨论"))
        thread = session["threads"][session["active_thread"]]
        self.assertEqual(thread["memory"]["revision"], 2)
        turns = harness.completed_turns(thread)
        self.assertEqual(thread["memory"]["covered_turn_ids"], [t["id"] for t in turns[:10]])
        sent = json.loads(self.requests[-1]["messages"][1]["content"])
        self.assertEqual(sent["previous_memory"]["summary"], self.content(summary="摘要一")["summary"])
        self.assertEqual([item["turn_id"] for item in sent["batch"]], [t["id"] for t in turns[5:10]])
        self.assertEqual(len(harness.context(session)), 14)

    def test_user_correction_in_recent_originals_then_updates_same_fact(self):
        questions = ["我喜欢37", "问2", "问3", "问4", "问5", "我改口了，我喜欢92", "问7", "问8", "问9", "问10"]
        session = self.seed(questions=questions)
        first = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([first]))
        messages = harness.context(session)
        self.assertIn("近期用户的明确纠正优先", messages[0]["content"])
        self.assertEqual(messages[2]["content"], questions[5])
        for i in range(5):
            self.add_chat(session, f"补充{i}")
        update = self.fact(harness.completed_turns(session["threads"][session["active_thread"]])[5], value="92")
        session = self.compact(session, self.content([update], summary="用户已明确将偏好从37改为92。"))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["facts"], [update])
        self.assertEqual(memory["revision"], 2)
        self.assertEqual(self.turns(session)[10]["memory_update"]["facts"], [first])

    def test_old_facts_cannot_disappear_or_change_without_new_evidence(self):
        session = self.seed()
        first = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([first]))
        for i in range(5):
            self.add_chat(session, f"补充{i}")
        old = copy.deepcopy(session["threads"][session["active_thread"]]["memory"])
        changed = copy.deepcopy(first)
        changed["value"] = "92"
        for data in (self.content(), self.content([changed])):
            with self.subTest(data=data), self.assertRaises(harness.MemoryError):
                self.compact(session, data)
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], old)
            self.assertEqual(harness.context(session)[2]["content"], "问题6")
            self.assertEqual(self.turns(session)[-1]["status"], "failed")

    def test_explicit_withdrawal_has_user_source_and_retains_old_revision(self):
        questions = ["我喜欢37", "问2", "问3", "问4", "问5", "请撤回我喜欢数字37这个偏好", "问7", "问8", "问9", "问10"]
        session = self.seed(questions=questions)
        session = self.compact(session, self.content([self.fact(self.turns(session)[0])]))
        for i in range(5):
            self.add_chat(session, f"补充{i}")
        source = self.fact(harness.completed_turns(session["threads"][session["active_thread"]])[5])["sources"]
        session = self.compact(session, self.content(removed=[{
            "key": "favorite_number", "reason": "用户明确撤回", "sources": source}]))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["facts"], [])
        self.assertEqual(memory["removed_facts"][0]["key"], "favorite_number")
        self.assertEqual(self.turns(session)[10]["memory_update"]["facts"][0]["value"], "37")

    def test_invalid_sources_and_assistant_guesses_are_rejected(self):
        session = self.seed()
        original = self.fact(self.turns(session)[0])
        cases = []
        for field, value in (("turn_id", "turn_000000000000"), ("turn_id", self.turns(session)[8]["id"]),
                             ("message_index", True), ("message_index", 999), ("quote", "不存在的原文")):
            fact = copy.deepcopy(original)
            fact["sources"][0][field] = value
            cases.append(fact)
        cases.append(self.fact(self.turns(session)[0], index=1))
        other = self.seed()
        cases.append(self.fact(self.turns(other)[0]))
        for fact in cases:
            with self.subTest(fact=fact), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content([fact]))
            memory = self.read(session["id"])["threads"][session["active_thread"]]["memory"]
            self.assertEqual(memory, harness.empty_memory())
            self.assertEqual(self.turns(session)[-1]["runs"][0]["usage"]["total_tokens"], 120)

    def test_summary_shape_limits_extra_fields_and_duplicates_fail_safely(self):
        session = self.seed()
        fact = self.fact(self.turns(session)[0])
        cases = [[], {}, self.content(summary=""), self.content(summary="长" * 3001),
                 self.content([fact, fact]), self.content([fact] * 33),
                 {**self.content(), "covered_turn_ids": []}]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(harness.MemoryError):
                self.compact(session, data)
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], harness.empty_memory())
        for text in ('{"facts":[],"facts":[],"summary":"摘要","removed_facts":[]}',
                     '{"facts":[],"summary":NaN,"removed_facts":[]}'):
            with self.assertRaises(harness.MemoryError):
                harness.decode_memory(text)

    def test_tool_evidence_is_validated_and_whole_tool_turns_remain(self):
        session = self.seed(tool_index=0)
        fact = self.fact(self.turns(session)[0], value="1081", key="calculation_result", index=2, kind="tool")
        session = self.compact(session, self.content([fact]))
        sent = json.loads(self.requests[-1]["messages"][1]["content"])
        self.assertEqual([m["role"] for m in sent["batch"][0]["messages"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(self.turns(session)[0]["messages"][2]["tool_call_id"], "call_calc_1")
        session2 = self.seed(tool_index=5)
        session2 = self.compact(session2, self.content())
        recent = harness.context(session2)[2:6]
        self.assertEqual([m["role"] for m in recent], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(recent[2]["tool_call_id"], "call_calc_1")

    def test_failed_tool_result_cannot_be_confirmed_fact(self):
        session = self.seed(tool_index=0)
        self.turns(session)[0]["messages"][2]["content"] = json.dumps({"ok": False, "error_type": "ToolError", "error": "失败"})
        self.store.save(session)
        fact = self.fact(self.turns(session)[0], value="计算成功", key="result", index=2, kind="tool")
        with self.assertRaises(harness.MemoryError):
            self.compact(session, self.content([fact]))

    def test_failed_chat_and_compaction_do_not_take_history_slots(self):
        session = self.seed(5)
        failed = harness.begin_turn(session, self.store, "失败")
        harness.finish_turn(session, self.store, failed, "failed", error_type="TimeoutError")
        self.add_chat(session, "第6轮")
        session = self.compact(session, self.content())
        thread = session["threads"][session["active_thread"]]
        self.assertEqual(len(thread["memory"]["covered_turn_ids"]), 1)
        self.assertEqual(len(harness.completed_turns(thread)), 6)
        self.assertNotIn("失败", [m["content"] for m in harness.context(session)[2:]])
        self.assertIn("已完成问答 6 轮", harness.threads_text(session))

    def test_http_empty_truncated_json_and_tool_call_failures_preserve_previous_memory(self):
        session = self.seed()
        first = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([first]))
        for i in range(5):
            self.add_chat(session, f"补充{i}")
        old = copy.deepcopy(session["threads"][session["active_thread"]]["memory"])
        errors = [TimeoutError(FAKE_KEY), HTTPError(main.API_URL, 401, FAKE_KEY, {"Authorization": FAKE_KEY}, None),
                  metered(reply("")), metered(reply("{bad" + FAKE_KEY)), metered(reply("{}", reason="length")),
                  metered(reply(None, [tool_call()]))]
        for response in errors:
            self.script = [response]
            with self.subTest(response=type(response).__name__), self.assertRaises(Exception):
                harness.run_compaction(session, self.store, lambda plan, rec:
                    main.summarize_history(plan, rec, FAKE_KEY, "offline-model"))
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], old)
            self.assertEqual(self.turns(session)[-1]["messages"], [])
            self.assertEqual(self.turns(session)[-1]["status"], "failed")
            self.assertNotIn(FAKE_KEY, self.store.path(session["id"]).read_text(encoding="utf-8"))
        self.clock.assert_not_called()
        self.weather_transport.assert_not_called()

    def test_cli_compaction_failure_keeps_loop_and_uncovered_history(self):
        session = self.seed()
        self.script = [metered(reply("{bad")), metered(reply("正常继续"))]
        status, output = self.run_cli(["/load " + session["id"], "/compact", "/memory", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("上一份有效记忆和全部未压缩原文保留", output)
        self.assertIn("正常继续", output)
        self.assertEqual(len(self.requests[-1]["messages"]), 22)
        self.assertEqual(self.requests[-1]["messages"][1]["content"], self.turns(session)[0]["input_text"])
        saved = self.read(session["id"])
        self.assertEqual(self.turns(saved)[-2]["status"], "failed")
        self.assertEqual(self.turns(saved)[-1]["kind"], "chat")

    def test_cli_compaction_success_next_request_uses_memory_and_only_uncovered_originals(self):
        session = self.seed()
        fact = self.fact(self.turns(session)[0])
        self.script = [self.response(self.content([fact])), metered(reply("你喜欢37"))]
        status, output = self.run_cli(["/load " + session["id"], "/compact", "/memory", "/context",
                                       "喜欢什么", "/runs", "/trace", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("记忆整理已保存", output)
        self.assertIn("记忆版本：1", output)
        self.assertIn("记忆覆盖：5轮", output)
        self.assertIn("类型：记忆整理", output)
        sent = self.requests[-1]["messages"]
        self.assertEqual(len(sent), 13)
        self.assertIn("37", sent[1]["content"])
        self.assertEqual(sent[2]["content"], "问题6")
        self.assertEqual(sent[-1]["content"], "喜欢什么")
        saved = self.read(session["id"])
        self.assertEqual(self.turns(saved)[-1]["messages"][0], {"role": "user", "content": "喜欢什么"})
        self.assertEqual(len(self.turns(saved)[-1]["messages"]), 2)

    def test_no_automatic_compaction_and_bad_command_arguments_stay_local(self):
        session = self.seed(12)
        self.script = [metered(reply("普通回答"))]
        status, output = self.run_cli(["/load " + session["id"], "/compact extra", "/memory extra", "/context extra",
                                       "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("response_format", self.requests[0])
        self.assertEqual(len(self.requests[0]["messages"]), 26)
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"], harness.empty_memory())

    def test_compaction_storage_failure_at_each_boundary_stops_without_commit(self):
        session = self.seed()
        sid = session["id"]
        response = self.response(self.content([self.fact(self.turns(session)[0])]))
        real_save = harness.Store.save
        pristine = self.store.path(sid).read_bytes()
        for failure, expected_calls in ((1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1)):
            self.store.path(sid).write_bytes(pristine)
            calls = []

            def save(store, candidate):
                calls.append(candidate["id"])
                if len(calls) == failure:
                    raise harness.StorageError("模拟整理保存失败，停止。")
                return real_save(store, candidate)

            self.requests.clear()
            self.script = [response]
            with self.subTest(failure=failure), patch.object(harness.Store, "save", save):
                status, output = self.run_cli(["/load " + sid, "/compact", "不应继续"])
            self.assertEqual(status, 1)
            self.assertEqual(len(self.requests), expected_calls)
            self.assertEqual(len(calls), failure)
            self.assertNotIn("记忆整理已保存", output)
            saved = self.read(sid)
            self.assertEqual(saved["threads"][saved["active_thread"]]["memory"], harness.empty_memory())
            if failure == 6:
                self.assertEqual(self.turns(saved)[-1]["status"], "running")
                loaded = self.store.load(sid)
                self.assertEqual(self.turns(loaded)[-1]["status"], "interrupted")
                self.assertEqual(self.turns(loaded)[-1]["runs"][0]["usage"]["total_tokens"], 120)

    def test_cancelled_compaction_retains_originals_and_is_not_replayed(self):
        session = self.seed()
        self.script = [KeyboardInterrupt()]
        status, output = self.run_cli(["/load " + session["id"], "/compact"])
        self.assertEqual(status, 130)
        loaded = self.store.load(session["id"])
        turn = self.turns(loaded)[-1]
        self.assertEqual(turn["status"], "interrupted")
        self.assertEqual(turn["runs"][0]["steps"][0]["status"], "interrupted")
        self.assertEqual(loaded["threads"][loaded["active_thread"]]["memory"], harness.empty_memory())
        self.assertEqual(len(self.requests), 1)

    def test_compaction_missing_or_partial_usage_is_not_reported_as_zero(self):
        for raw, state in ((None, "missing"), ({"prompt_tokens": 100}, "partial"),
                           ({"total_tokens": True}, "invalid")):
            session = self.seed()
            response = reply(json.dumps(self.content(), ensure_ascii=False))
            if raw is not None:
                response["usage"] = raw
            self.script = [response]
            with self.subTest(state=state):
                session, changed = harness.run_compaction(session, self.store, lambda plan, rec:
                    main.summarize_history(plan, rec, FAKE_KEY, "offline-model"))
                self.assertTrue(changed)
                run = self.turns(session)[-1]["runs"][0]
                self.assertEqual(run["steps"][0]["usage"]["state"], state)
                self.assertEqual(run["usage"]["model_calls"], 1)
                self.assertFalse(run["usage"]["complete"])
                self.assertIsNone(run["usage"]["total_tokens"])
                self.assertIn("总计 未知", harness.run_summary(run))
                self.assertEqual(self.store.load(session["id"]), session)

    def test_running_compaction_in_inactive_thread_recovers_without_replay(self):
        session = self.seed()
        a = session["active_thread"]
        plan = harness.prepare_compaction(session)
        turn = harness.begin_turn(session, self.store, "/compact", traced=True, kind="compaction")
        recorder = harness.Recorder(session, self.store, turn)
        recorder.start("llm", main.compaction_payload(plan, "offline-model"))
        session = harness.new_thread(session, self.store)
        b = session["active_thread"]
        loaded = self.store.load(session["id"])
        self.assertEqual(loaded["active_thread"], b)
        saved = loaded["threads"][a]["turns"][-1]
        self.assertEqual(saved["status"], "interrupted")
        self.assertTrue(all(event["thread_id"] == a for event in saved["runs"][0]["events"]))
        self.assertEqual(loaded["threads"][a]["memory"], harness.empty_memory())
        self.assertEqual(len(harness.completed_turns(loaded["threads"][a])), 10)
        self.assertEqual(harness.context(loaded), [{"role": "system", "content": "共同系统提示"}])
        self.transport.assert_not_called()

    def test_memory_survives_failed_followup_and_reload(self):
        session = self.seed()
        fact = self.fact(self.turns(session)[0])
        self.script = [self.response(self.content([fact])), TimeoutError(FAKE_KEY)]
        self.run_cli(["/load " + session["id"], "/compact", "失败问题", "exit"])
        loaded = self.store.load(session["id"])
        self.assertEqual(loaded["threads"][loaded["active_thread"]]["memory"]["revision"], 1)
        self.assertEqual(self.turns(loaded)[-1]["status"], "failed")
        self.requests.clear()
        self.script = [metered(reply("37"))]
        status, output = self.run_cli(["/load " + session["id"], "/memory", "恢复后的问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("37", self.requests[-1]["messages"][1]["content"])
        self.assertNotIn("失败问题", [m["content"] for m in self.requests[-1]["messages"]])

    def test_memories_are_isolated_and_new_thread_starts_empty(self):
        session = self.seed()
        a = session["active_thread"]
        session = self.compact(session, self.content([self.fact(self.turns(session)[0])]))
        a_memory = copy.deepcopy(session["threads"][a]["memory"])
        session = harness.new_thread(session, self.store)
        b = session["active_thread"]
        self.assertEqual(session["threads"][b]["memory"], harness.empty_memory())
        self.assertEqual(len(harness.context(session)), 1)
        for i in range(10):
            self.add_chat(session, "我喜欢92" if i == 0 else f"B问题{i}")
        session = self.compact(session, self.content([self.fact(self.turns(session)[0], value="92")]))
        self.assertEqual(session["threads"][a]["memory"], a_memory)
        self.assertIn('"value": "92"', harness.context(session)[1]["content"])
        session = harness.use_thread(session, self.store, a)
        self.assertIn('"value": "37"', harness.context(session)[1]["content"])
        self.assertNotIn('"value": "92"', harness.context(session)[1]["content"])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_large_request_is_stopped_without_truncating_history(self):
        session = self.seed()
        self.add_chat(session, "长" * (harness.MAX_REQUEST_BYTES // 2))
        before = copy.deepcopy(self.turns(session))
        status, output = self.run_cli(["/load " + session["id"], "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("128 KiB", output)
        self.transport.assert_not_called()
        saved = self.read(session["id"])
        self.assertEqual(self.turns(saved)[:-1], before)
        self.assertEqual(self.turns(saved)[-1]["status"], "failed")
        self.assertEqual(self.turns(saved)[-1]["runs"][0]["usage"]["model_calls"], 0)

    def test_large_compaction_input_fails_without_request_or_advancing_coverage(self):
        session = self.seed()
        self.turns(session)[0]["input_text"] = "长" * (harness.MAX_REQUEST_BYTES // 2)
        self.turns(session)[0]["messages"][0]["content"] = self.turns(session)[0]["input_text"]
        self.store.save(session)
        with self.assertRaises(harness.ContextError):
            harness.run_compaction(session, self.store, lambda plan, rec:
                main.summarize_history(plan, rec, FAKE_KEY, "offline-model"))
        self.transport.assert_not_called()
        self.assertEqual(session["threads"][session["active_thread"]]["memory"], harness.empty_memory())

    def test_v3_multithread_upgrade_keeps_records_creates_empty_memory_and_backup(self):
        session = self.seed(2)
        session = harness.new_thread(session, self.store)
        self.add_chat(session, "B问题")
        old = as_legacy(session, 3)
        self.store.save(old)
        target = self.store.path(old["id"])
        original = target.read_bytes()
        loaded = self.store.load(old["id"])
        self.assertEqual(loaded, expected_v4(old))
        self.assertEqual(target.with_suffix(".v3.bak.json").read_bytes(), original)
        self.assertEqual(harness.context(loaded), harness.context(old))
        self.assertEqual(self.store.load(old["id"]), loaded)
        self.transport.assert_not_called()

    def test_v3_backup_or_upgrade_write_failure_keeps_original(self):
        old = as_legacy(self.seed(2), 3)
        self.store.save(old)
        target = self.store.path(old["id"])
        original = target.read_bytes()
        for function in ("rename", "replace"):
            with self.subTest(function=function), patch.object(harness.os, function, side_effect=OSError(FAKE_KEY)):
                with self.assertRaises(harness.StorageError):
                    self.store.load(old["id"])
            self.assertEqual(target.read_bytes(), original)
        self.assertEqual(target.with_suffix(".v3.bak.json").read_bytes(), original)
        self.assertEqual(self.store.load(old["id"])["version"], 4)

    def test_corrupt_memory_coverage_revision_sources_and_task_types_rejected(self):
        session = self.seed()
        session = self.compact(session, self.content([self.fact(self.turns(session)[0])]))
        changes = [lambda s: s["threads"][s["active_thread"]]["memory"].update(revision=99),
                   lambda s: self.turns(s)[-1].update(kind="chat"),
                   lambda s: self.turns(s)[-1]["runs"][0].update(task_type="chat"),
                   lambda s: self.turns(s)[-1]["memory_update"]["covered_turn_ids"].reverse(),
                   lambda s: self.turns(s)[-1]["memory_update"]["facts"][0]["sources"][0].update(message_index=1),
                   lambda s: self.turns(s)[-1]["runs"][0]["steps"][0]["input_data"].update(tools=[])]
        target = self.store.path(session["id"])
        for index, change in enumerate(changes):
            bad = copy.deepcopy(session)
            change(bad)
            raw = json.dumps(bad).encode("utf-8")
            target.write_bytes(raw)
            with self.subTest(index=index), self.assertRaises(harness.SessionError):
                self.store.load(session["id"])
            self.assertEqual(target.read_bytes(), raw)

    def test_compaction_record_must_match_actual_batch_and_saved_output(self):
        session = self.seed()
        session = self.compact(session, self.content([self.fact(self.turns(session)[0])]))
        for change in ("input_batch", "previous_summary", "output_summary"):
            bad = copy.deepcopy(session)
            step = self.turns(bad)[-1]["runs"][0]["steps"][0]
            if change == "output_summary":
                data = json.loads(step["output_data"]["content"])
                data["summary"] = "另一份摘要"
                step["output_data"]["content"] = json.dumps(data)
            else:
                data = json.loads(step["input_data"]["messages"][1]["content"])
                if change == "input_batch":
                    data["batch"][0]["messages"][0]["content"] = "与原文不同"
                else:
                    data["previous_memory"]["summary"] = "不存在的上一份摘要"
                step["input_data"]["messages"][1]["content"] = json.dumps(data)
            with self.subTest(change=change), self.assertRaises(harness.SessionError):
                harness.validate_session(bad)

    def test_three_compactions_keep_learning_topics_through_confirmation_batch(self):
        questions = ["我喜欢的数字是37。这是我的明确偏好，请记住。",
                     "解释变量", "解释列表", "解释函数", "解释循环", "解释API"]
        session = self.seed(6, questions=questions)
        original = copy.deepcopy(self.turns(session))
        fact = self.fact(original[0])
        topics = "变量、列表、函数、循环和API"
        summaries = [self.structured_summary(topic="记录用户偏好", completed="已记录数字37",
                                              pending="用户偏好数字37"),
                     self.structured_summary(completed="已学习" + topics, pending="用户偏好数字37"),
                     self.structured_summary(completed="已学习" + topics,
                                              state="本批仅确认既有数字偏好", pending="用户偏好数字37")]
        session = self.compact(session, self.content([fact], summaries[0]))
        self.assertEqual(len(harness.prepare_compaction(session)["previous_memory"]["covered_turn_ids"]), 1)
        for question in ("我喜欢的数字是什么？", "你好呀", "嘻嘻", "你是谁", "聊一聊", "再见"):
            self.add_chat(session, question)
        for index in (1, 2):
            session = self.compact(session, self.content([fact], summaries[index]))
            sent = json.loads(self.requests[-1]["messages"][1]["content"])
            self.assertEqual(sent["previous_memory"]["summary"], summaries[index - 1])
        thread = session["threads"][session["active_thread"]]
        self.assertEqual(thread["memory"]["revision"], 3)
        self.assertEqual(len(thread["memory"]["covered_turn_ids"]), 7)
        self.assertEqual(self.turns(session)[:6], original)
        final_batch = json.loads(self.requests[-1]["messages"][1]["content"])["batch"]
        self.assertEqual(len(final_batch), 1)
        self.assertEqual(final_batch[0]["messages"][0]["content"], "我喜欢的数字是什么？")
        self.assertIn(topics, thread["memory"]["summary"])
        self.assertEqual(thread["memory"]["facts"], [fact])
        self.assertEqual(len(harness.completed_turns(thread)) - 7, 5)
        self.assertEqual(self.store.load(session["id"]), session)
        self.script = [metered(reply("已学过" + topics))]
        status, output = self.run_cli(["/load " + session["id"], "我们之前学过什么？", "exit"])
        self.assertEqual(status, 0)
        sent = self.requests[-1]["messages"]
        self.assertIn(topics, sent[1]["content"])
        self.assertEqual([m["content"] for m in sent[2:-1:2]],
                         ["你好呀", "嘻嘻", "你是谁", "聊一聊", "再见"])
        self.assertNotIn("解释变量", [m["content"] for m in sent[2:]])

    def test_unchanged_old_goals_and_pending_items_pass_to_later_compaction(self):
        session = self.seed(16)
        summary = self.structured_summary(topic="逐阶段学习Agent Harness",
            completed="已解释变量与列表", state="等待用户完成练习",
            pending="用户需要完成变量练习；保持使用Python标准库")
        session = self.compact(session, self.content(summary=summary))
        session = self.compact(session, self.content(summary=summary))
        sent = json.loads(self.requests[-1]["messages"][1]["content"])
        self.assertEqual(sent["previous_memory"]["summary"], summary)
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["summary"], summary)
        self.assertEqual(len(harness.prepare_compaction(session)["previous_memory"]["covered_turn_ids"]), 10)

    def test_structured_summary_accepts_explicit_goal_completion_and_preference_update(self):
        questions = ["我喜欢37", "解释变量", "我准备做变量练习", "解释列表", "解释函数",
                     "变量练习已经完成；我改口了，现在喜欢92", "问7", "问8", "问9", "问10"]
        session = self.seed(questions=questions)
        session = self.compact(session, self.content([self.fact(self.turns(session)[0])],
            self.structured_summary(completed="已学习变量、列表和函数", pending="待完成变量练习；偏好37")))
        for i in range(5):
            self.add_chat(session, f"补充问题{i}")
        source = harness.completed_turns(session["threads"][session["active_thread"]])[5]
        summary = self.structured_summary(completed="已学习变量、列表和函数；用户确认已完成变量练习",
                                         state="等待新的学习任务", pending="偏好已改为92；无其他待办")
        session = self.compact(session, self.content([self.fact(source, value="92")], summary))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["summary"], summary)
        self.assertEqual(memory["facts"][0]["value"], "92")
        self.assertEqual(memory["facts"][0]["sources"][0]["turn_id"], source["id"])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_missing_reordered_duplicate_or_empty_sections_fail_without_advancing_memory(self):
        session = self.seed(16)
        fact = self.fact(self.turns(session)[0])
        valid = self.structured_summary(completed="已学过变量与列表", pending="还需完成练习；偏好37")
        session = self.compact(session, self.content([fact], valid))
        old = copy.deepcopy(session["threads"][session["active_thread"]]["memory"])
        before_context = harness.context(session)
        sections = valid.splitlines()
        cases = ["只写本批确认数字37", "\n".join(sections[:-1]), "\n".join(sections[::-1]),
                 valid + "\n主题与目标：重复", valid.replace(sections[1], "已完成内容："),
                 "摘要如下：\n" + valid, None]
        for summary in cases:
            data = {"facts": [fact], "summary": summary, "removed_facts": []}
            with self.subTest(summary=summary), self.assertRaises(harness.MemoryError):
                self.compact(session, data)
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], old)
            self.assertEqual(harness.context(session), before_context)
            run = self.turns(session)[-1]["runs"][0]
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["steps"][0]["status"], "failed")
            self.assertEqual(run["usage"]["total_tokens"], 120)
            self.assertEqual(self.store.load(session["id"]), session)

    def test_invalid_sections_cli_keeps_history_and_allows_next_question(self):
        session = self.seed()
        data = {"facts": [], "summary": "只写本批内容", "removed_facts": []}
        self.script = [self.response(data), metered(reply("继续回答"))]
        status, output = self.run_cli(["/load " + session["id"], "/compact", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("滚动摘要必须按顺序包含", output)
        self.assertIn("继续回答", output)
        self.assertEqual(len(self.requests[-1]["messages"]), 22)
        self.assertEqual(self.read(session["id"])["threads"][session["active_thread"]]["memory"], harness.empty_memory())

    def test_existing_v4_freeform_summary_loads_unchanged_then_can_be_compacted(self):
        session = self.seed()
        original_summary = "喜欢数字37；已讨论变量、列表、函数、循环和API；待完成练习。"
        data = {"facts": [self.fact(self.turns(session)[0])], "summary": original_summary, "removed_facts": []}

        def old_runner(plan, recorder):
            with patch.object(main, "COMPACTION_INSTRUCTIONS", "旧版记忆整理器，输出json。"):
                payload = main.compaction_payload(plan, "offline-model")
            step = recorder.start("llm", payload)
            recorder.usage(step, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
            recorder.finish(step, {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            return harness.build_memory(plan, data)

        session, _ = harness.run_compaction(session, self.store, old_runner)
        target = self.store.path(session["id"])
        before = target.read_bytes()
        loaded = self.store.load(session["id"])
        self.assertEqual(loaded, session)
        self.assertEqual(target.read_bytes(), before)
        self.transport.assert_not_called()
        for i in range(5):
            self.add_chat(loaded, f"后续问题{i}")
        summary = self.structured_summary(completed="已讨论变量、列表、函数、循环和API", pending="待完成练习；偏好37")
        session = self.compact(loaded, self.content(data["facts"], summary))
        sent = json.loads(self.requests[-1]["messages"][1]["content"])
        self.assertEqual(sent["previous_memory"]["summary"], original_summary)
        self.assertEqual(session["version"], 4)
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["summary"], summary)
        self.assertEqual(self.turns(session)[10]["memory_update"]["summary"], original_summary)
        self.assertEqual(self.store.load(session["id"]), session)

    def test_new_format_marker_checks_saved_format_even_if_all_copies_match(self):
        session = self.compact(self.seed(), self.content())
        turn = self.turns(session)[-1]
        turn["memory_update"]["summary"] = "省略四个部分"
        session["threads"][session["active_thread"]]["memory"]["summary"] = "省略四个部分"
        output = turn["runs"][0]["steps"][0]["output_data"]
        data = json.loads(output["content"])
        data["summary"] = "省略四个部分"
        output["content"] = json.dumps(data, ensure_ascii=False)
        target = self.store.path(session["id"])
        raw = json.dumps(session).encode("utf-8")
        target.write_bytes(raw)
        with self.assertRaises(harness.SessionError):
            self.store.load(session["id"])
        self.assertEqual(target.read_bytes(), raw)

    def legacy_memory(self, session, facts, summary=None):
        """构造旧v4记录；无新出处标记，不读取任何真实会话。"""
        data = {"facts": facts, "summary": summary or self.structured_summary(completed="助手已解释变量"),
                "removed_facts": []}

        def runner(plan, recorder):
            with patch.object(main, "COMPACTION_INSTRUCTIONS", "旧版json整理。" + harness.SUMMARY_FORMAT_MARKER):
                payload = main.compaction_payload(plan, "offline-model")
            step = recorder.start("llm", payload)
            recorder.usage(step, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
            recorder.finish(step, {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            return harness.build_memory(plan, data)

        return harness.run_compaction(session, self.store, runner)[0]

    def test_user_question_cannot_support_assistant_definition_or_mastery_fact(self):
        session = self.seed(questions=["请用一句话解释变量。"] + [f"问题{i}" for i in range(9)])
        before = harness.context(session)
        for value in ("已向用户解释变量：变量是给值起名", "用户已经学会变量"):
            fact = self.fact(self.turns(session)[0], value=value, key="learned_variables")
            with self.subTest(value=value), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content([fact]))
            self.assertEqual(harness.context(session), before)
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], harness.empty_memory())
            self.assertEqual(self.turns(session)[-1]["runs"][0]["usage"]["total_tokens"], 120)

    def test_user_self_report_is_stored_as_literal_report_not_independent_verification(self):
        session = self.seed(questions=["我已经完成变量练习"] + [f"问题{i}" for i in range(9)])
        fact = self.fact(self.turns(session)[0], value="我已经完成变量练习", key="reported_progress")
        summary = self.structured_summary(completed="用户自述已经完成变量练习；未独立测试掌握程度")
        session = self.compact(session, self.content([fact], summary))
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["facts"], [fact])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_preference_question_cannot_be_added_as_confirmation_source(self):
        questions = ["我喜欢的数字是37", "问2", "问3", "问4", "问5", "我喜欢的数字是什么？",
                     "问7", "问8", "问9", "问10"]
        session = self.seed(questions=questions)
        fact = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([fact]))
        for i in range(5):
            self.add_chat(session, f"补充{i}")
        changed = copy.deepcopy(fact)
        question = harness.completed_turns(session["threads"][session["active_thread"]])[5]
        changed["sources"].extend(self.fact(question)["sources"])
        old = copy.deepcopy(session["threads"][session["active_thread"]]["memory"])
        with self.assertRaises(harness.MemoryError):
            self.compact(session, self.content([changed]))
        self.assertEqual(session["threads"][session["active_thread"]]["memory"], old)

    def test_wish_and_bare_acknowledgment_do_not_form_specific_agreement(self):
        for text in ("希望你也开心", "好的"):
            session = self.seed(questions=[text] + [f"问题{i}" for i in range(9)])
            fact = self.fact(self.turns(session)[0], value=text, key="user_statement")
            summary = self.structured_summary(suggestions="助手建议每天记录一件开心小事",
                                              agreements=[text])
            with self.subTest(text=text), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content([fact], summary))
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], harness.empty_memory())

    def test_assistant_suggestion_stays_unconfirmed_without_user_acceptance(self):
        session = self.seed(questions=["我希望每天开心"] + [f"问题{i}" for i in range(9)])
        self.turns(session)[0]["messages"][-1]["content"] = "我建议每天记录一件开心小事，那就说定了。"
        self.store.save(session)
        fact = self.fact(self.turns(session)[0], value="我希望每天开心", key="daily_happiness_wish")
        summary = self.structured_summary(suggestions="助手建议每天记录一件开心小事；用户尚未确认")
        session = self.compact(session, self.content([fact], summary))
        self.assertIn("用户明确约定：无", session["threads"][session["active_thread"]]["memory"]["summary"])
        bad_summary = self.structured_summary(agreements=["我同意每天记录一件开心小事"])
        with self.assertRaises(harness.MemoryError):
            harness.check_fact_evidence(harness.prepare_compaction(session), self.content([fact], bad_summary))

    def test_explicit_complete_user_acceptance_is_recorded_when_its_batch_is_compacted(self):
        accepted = "我同意每天做一个变量练习"
        questions = ["我想学习Python", "问2", "问3", "问4", "问5", accepted, "问7", "问8", "问9", "问10"]
        session = self.seed(questions=questions)
        goal = self.fact(self.turns(session)[0], value="我想学习Python", key="learning_goal")
        session = self.compact(session, self.content([goal], self.structured_summary(suggestions="每天做一个变量练习")))
        for i in range(5):
            self.add_chat(session, f"补充{i}")
        source = harness.completed_turns(session["threads"][session["active_thread"]])[5]
        fact = self.fact(source, value=accepted, key="daily_variable_practice")
        summary = self.structured_summary(pending="每天做一个变量练习", agreements=[accepted])
        session = self.compact(session, self.content([goal, fact], summary))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertIn(json.dumps([accepted], ensure_ascii=False), memory["summary"])
        self.assertEqual(memory["facts"][-1]["sources"][0]["turn_id"], source["id"])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_agreement_cannot_be_cherry_picked_from_hypothetical_or_quoted_message(self):
        accepted = "我同意每天做一个变量练习"
        for text in ("如果" + accepted + "，会有什么变化？", "我不认为“" + accepted + "”这句话是我的决定"):
            session = self.seed(questions=[text] + [f"问题{i}" for i in range(9)])
            fact = self.fact(self.turns(session)[0], value=accepted, key="practice_agreement")
            with self.subTest(text=text), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content([fact], self.structured_summary(agreements=[accepted])))

    def test_complete_acceptance_may_omit_terminal_period_but_not_be_a_question(self):
        accepted = "我同意每天做一个变量练习"
        session = self.seed(questions=[accepted + "。"] + [f"问题{i}" for i in range(9)])
        fact = self.fact(self.turns(session)[0], value=accepted, key="practice_agreement")
        session = self.compact(session, self.content([fact], self.structured_summary(agreements=[accepted])))
        self.assertEqual(self.store.load(session["id"]), session)
        for question in (accepted + "吗？", accepted + "?"):
            session = self.seed(questions=[question] + [f"问题{i}" for i in range(9)])
            fact = self.fact(self.turns(session)[0], value=question, key="practice_agreement")
            with self.subTest(question=question), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content([fact], self.structured_summary(agreements=[question])))

    def test_legacy_unsupported_learning_fact_moves_to_summary_with_correction_record(self):
        questions = ["我喜欢37", "请解释变量", "问3", "问4", "问5"] + [f"问题{i}" for i in range(11)]
        session = self.seed(16, questions=questions)
        preference = self.fact(self.turns(session)[0])
        learned = self.fact(self.turns(session)[1], value="已向用户解释变量：变量是给值起名", key="learned_variables")
        session = self.legacy_memory(session, [preference, learned])
        path = self.store.path(session["id"])
        before = path.read_bytes()
        self.assertEqual(self.store.load(session["id"]), session)
        self.assertEqual(path.read_bytes(), before)
        correction = {"key": "learned_variables", "reason": "出处纠正：用户问题不能作为助手定义的来源，移回进展摘要",
                      "sources": copy.deepcopy(learned["sources"])}
        session = self.compact(session, self.content([preference],
            self.structured_summary(completed="助手已解释变量，未独立验证用户掌握"), removed=[correction]))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["facts"], [preference])
        self.assertEqual(memory["removed_facts"], [correction])
        self.assertEqual(self.turns(session)[16]["memory_update"]["facts"], [preference, learned])
        self.assertIn("变量", memory["summary"])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_legacy_abstract_goal_can_be_corrected_to_original_user_words(self):
        session = self.seed(16, questions=["我想学习Python"] + [f"问题{i}" for i in range(15)])
        abstract = self.fact(self.turns(session)[0], value="用户请求学习Python", key="learning_goal")
        session = self.legacy_memory(session, [abstract])
        literal = copy.deepcopy(abstract)
        literal["value"] = "我想学习Python"
        session = self.compact(session, self.content([literal]))
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["facts"], [literal])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_legacy_weak_question_source_can_be_removed_without_changing_preference(self):
        questions = ["我喜欢37", "我喜欢的数字是什么？"] + [f"问题{i}" for i in range(14)]
        session = self.seed(16, questions=questions)
        good = self.fact(self.turns(session)[0])
        weak = copy.deepcopy(good)
        weak["sources"].extend(self.fact(self.turns(session)[1])["sources"])
        session = self.legacy_memory(session, [weak])
        session = self.compact(session, self.content([good]))
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["facts"], [good])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_provenance_correction_cannot_delete_or_change_valid_preference(self):
        session = self.seed(16)
        good = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([good]))
        old = copy.deepcopy(session["threads"][session["active_thread"]]["memory"])
        removed = {"key": good["key"], "reason": "出处纠正：删除", "sources": copy.deepcopy(good["sources"])}
        changed = copy.deepcopy(good)
        changed["value"] = "92"
        for data in (self.content(removed=[removed]), self.content([changed])):
            with self.subTest(data=data), self.assertRaises(harness.MemoryError):
                self.compact(session, data)
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], old)

    def test_provenance_correction_cannot_invent_old_sources(self):
        session = self.seed(16)
        bad = self.fact(self.turns(session)[0], value="用户喜欢的数字是37", key="old_preference")
        session = self.legacy_memory(session, [bad])
        invented = self.fact(harness.completed_turns(session["threads"][session["active_thread"]])[1])["sources"]
        removed = {"key": bad["key"], "reason": "出处纠正：没有出处", "sources": invented}
        with self.assertRaises(harness.MemoryError):
            self.compact(session, self.content(removed=[removed]))

    def test_malformed_agreement_sections_or_lists_fail_safely(self):
        session = self.seed()
        valid = self.structured_summary()
        cases = [valid.replace("用户明确约定：无", "用户明确约定：[]"),
                 valid.replace("用户明确约定：无", "用户明确约定：{bad"),
                 valid.replace("用户明确约定：无", "用户明确约定：[\"我同意每天练习\",\"我同意每天练习\"]"),
                 valid.replace("助手建议（未确认）：无\n", ""), valid + "\n额外说明：无"]
        for summary in cases:
            with self.subTest(summary=summary), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content(summary=summary))
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], harness.empty_memory())

    def test_invalid_fact_evidence_cli_keeps_loop_and_shows_no_response_secret(self):
        session = self.seed()
        fact = self.fact(self.turns(session)[0], value="无法由用户原话证明的结论" + FAKE_KEY)
        self.script = [self.response(self.content([fact])), metered(reply("继续回答"))]
        status, output = self.run_cli(["/load " + session["id"], "/compact", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("用户事实必须使用用户原话摘录", output)
        self.assertIn("继续回答", output)
        self.assertNotIn(FAKE_KEY, output)
        self.assertNotIn(FAKE_KEY, self.store.path(session["id"]).read_text(encoding="utf-8"))
        self.assertEqual(len(self.requests[-1]["messages"]), 22)

    def test_new_evidence_marker_rechecks_saved_user_fact_even_if_copies_match(self):
        session = self.seed()
        session = self.compact(session, self.content([self.fact(self.turns(session)[0])]))
        turn = self.turns(session)[-1]
        for memory in (turn["memory_update"], session["threads"][session["active_thread"]]["memory"]):
            memory["facts"][0]["value"] = "用户已经学会变量"
        output = turn["runs"][0]["steps"][0]["output_data"]
        data = json.loads(output["content"])
        data["facts"][0]["value"] = "用户已经学会变量"
        output["content"] = json.dumps(data, ensure_ascii=False)
        target = self.store.path(session["id"])
        raw = json.dumps(session).encode("utf-8")
        target.write_bytes(raw)
        with self.assertRaises(harness.SessionError):
            self.store.load(session["id"])
        self.assertEqual(target.read_bytes(), raw)

    def test_old_fact_shorter_excerpt_keeps_original_fact_and_compacts_next_batch(self):
        questions = ["我喜欢的数字是37，请记住。"] + [f"解释概念{i}" for i in range(15)]
        session = self.seed(16, questions=questions)
        original = self.fact(self.turns(session)[0], value="我喜欢的数字是37")
        session = self.compact(session, self.content([original]))
        copied = copy.deepcopy(original)
        copied["value"] = "37"
        session = self.compact(session, self.content([copied]))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["facts"], [original])
        self.assertEqual(memory["revision"], 2)
        self.assertEqual(len(memory["covered_turn_ids"]), 10)
        raw = json.loads(self.turns(session)[-1]["runs"][0]["steps"][0]["output_data"]["content"])
        self.assertEqual(raw["facts"], [copied])
        self.assertIn("输出是模型原始候选", harness.trace_text(session))
        self.assertEqual(self.store.load(session["id"]), session)

    def test_old_fact_shorter_source_quote_keeps_original_source(self):
        session = self.seed(16)
        original = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([original]))
        copied = copy.deepcopy(original)
        copied["sources"][0]["quote"] = "我喜欢的数字是37"
        session = self.compact(session, self.content([copied]))
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["facts"], [original])
        self.assertEqual(self.store.load(session["id"]), session)

    def test_old_fact_copy_cannot_hide_invalid_source_or_extra_field(self):
        session = self.seed(16)
        original = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([original]))
        old = copy.deepcopy(session["threads"][session["active_thread"]]["memory"])
        for change in ("quote", "assistant", "extra", "value"):
            copied = copy.deepcopy(original)
            if change == "quote":
                copied["sources"][0]["quote"] = "不在原文中的引文37"
            elif change == "assistant":
                copied["sources"][0].update(message_index=1, quote=self.turns(session)[0]["messages"][1]["content"])
            elif change == "extra":
                copied["unexpected"] = "不应被归一化抹去"
            else:
                copied["value"] = "92"
            with self.subTest(change=change), self.assertRaises(harness.MemoryError):
                self.compact(session, self.content([copied]))
            self.assertEqual(session["threads"][session["active_thread"]]["memory"], old)
            self.assertEqual(self.store.load(session["id"]), session)

    def test_renaming_old_fact_without_new_evidence_reports_safe_field_name(self):
        session = self.seed(16)
        original = self.fact(self.turns(session)[0])
        session = self.compact(session, self.content([original]))
        renamed = copy.deepcopy(original)
        renamed["key"] = "preferred_number"
        with self.assertRaises(harness.MemoryError) as caught:
            self.compact(session, self.content([renamed]))
        self.assertIn("问题字段：preferred_number（新增事实）", str(caught.exception))
        self.assertEqual(session["threads"][session["active_thread"]]["memory"]["facts"], [original])

    def test_reported_sixteen_turn_flow_compacts_copied_preference_then_real_agreement(self):
        questions = ["我喜欢的数字是37，请记住。", "请解释变量", "请建议每天记录开心小事",
                     "希望你也开心", "好的，谢谢", "请解释列表", "请解释函数", "请解释循环",
                     "请解释API", "请解释注释"]
        session = self.seed(questions=questions)
        original = self.fact(self.turns(session)[0], value="我喜欢的数字是37")
        first = self.structured_summary(completed="助手已解释变量", suggestions="每天记录一件开心小事")
        session = self.compact(session, self.content([original], first))
        accepted = "我同意每天记录一件开心小事"
        for text in (accepted, "请解释字符串", "请解释字典", "请解释布尔值", "请解释缩进", "请解释参数"):
            self.add_chat(session, text)
        copied = copy.deepcopy(original)
        copied["value"] = "37"
        second = self.structured_summary(completed="助手已解释变量、列表、函数、循环、API和注释",
                                         suggestions="每天记录一件开心小事")
        session = self.compact(session, self.content([copied], second))
        self.assertEqual(len(session["threads"][session["active_thread"]]["memory"]["covered_turn_ids"]), 10)
        source = harness.completed_turns(session["threads"][session["active_thread"]])[10]
        agreement = self.fact(source, value=accepted, key="daily_happiness_agreement")
        third = self.structured_summary(completed="助手已解释变量、列表、函数、循环、API和注释，未测试掌握程度",
                                       pending="每天记录一件开心小事", agreements=[accepted])
        session = self.compact(session, self.content([copied, agreement], third))
        memory = session["threads"][session["active_thread"]]["memory"]
        self.assertEqual(memory["revision"], 3)
        self.assertEqual(len(memory["covered_turn_ids"]), 11)
        self.assertEqual(memory["facts"], [original, agreement])
        self.assertEqual(len(harness.completed_turns(session["threads"][session["active_thread"]])), 16)
        self.assertIn("尚未压缩原文：5轮", harness.context_text(session))
        self.assertEqual(self.store.load(session["id"]), session)
        self.script = [metered(reply("喜欢37；同意记录开心小事；未验证掌握变量"))]
        status, output = self.run_cli(["/load " + session["id"], "检查数字和约定", "exit"])
        self.assertEqual(status, 0)
        self.assertIn(accepted, self.requests[-1]["messages"][1]["content"])
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"][2:-1:2]],
                         ["请解释字符串", "请解释字典", "请解释布尔值", "请解释缩进", "请解释参数"])


class AutoCompactionTests(OfflineCase):
    """实际 CLI/模拟传输验收；只使用临时会话，不改变正式阈值。"""

    ids = SessionTests.ids
    read = SessionTests.read
    turns = SessionTests.turns
    add_chat = MemoryTests.add_chat
    seed = MemoryTests.seed
    fact = MemoryTests.fact
    content = MemoryTests.content
    structured_summary = MemoryTests.structured_summary
    response = MemoryTests.response

    def setUp(self):
        super().setUp()
        self.store = harness.Store(self.session_root)

    def large_history(self, count=10, answer_chars=1500):
        session = self.seed(count)
        for turn in self.turns(session):
            turn["messages"][-1]["content"] = "旧回答：" + "资" * answer_chars
        self.store.save(session)
        return session

    def payload(self, session, question="新问题"):
        return main.model_payload(harness.context(session) + [{"role": "user", "content": question}], "offline-model")

    def test_size_uses_actual_utf8_body_with_tools_model_and_question(self):
        session = self.seed(1)
        payload = self.payload(session, "中文问题" * 10)
        raw = harness.check_request_size(payload)
        self.assertEqual(harness.request_size(payload), len(raw))
        self.assertEqual(raw, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        self.assertGreater(len(raw), len(raw.decode("utf-8")))
        self.assertGreater(harness.request_size(payload), harness.request_size({"messages": payload["messages"]}))
        self.script = [reply("离线回答")]
        self.run_turn("中文问题" * 10, harness.context(session))
        self.assertEqual(self.requests[0], payload)
        with self.assertRaises(ValueError):
            harness.request_size({"value": float("nan")})

    def test_switches_and_context_are_local_and_never_save_policy(self):
        session = self.large_history()
        before = self.store.path(session["id"]).read_bytes()
        status, output = self.run_cli(["/load " + session["id"], "/autocompact", "/autocompact on",
                                       "", "/context", "/memory", "/runs", "/trace", "help",
                                       "/autocompact bad", "/autocompact on extra", "/autocompact off", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("自动整理：关闭", output)
        self.assertIn("自动整理：开启", output)
        self.assertIn("尚未包含下一条问题", output)
        self.assertIn("32 KiB", output)
        self.assertIn("每个新问题最多自动整理2批", output)
        self.assertIn("命令或参数不正确", output)
        self.assertEqual(self.store.path(session["id"]).read_bytes(), before)
        self.transport.assert_not_called()
        status, output = self.run_cli(["/load " + session["id"], "/autocompact", "quit"])
        self.assertEqual(status, 0)
        self.assertNotIn("自动整理：开启", output)

    def test_default_off_sends_full_history_even_above_soft_threshold(self):
        session = self.large_history()
        self.assertGreater(harness.request_size(self.payload(session)), harness.AUTOCOMPACT_SOFT_BYTES)
        self.script = [reply("普通回答")]
        self.run_cli(["/load " + session["id"], "新问题", "exit"])
        self.assertEqual(self.requests, [self.payload(session)])
        self.assertEqual(self.read(session["id"])["threads"][session["active_thread"]]["memory"], harness.empty_memory())

    def test_at_or_below_soft_threshold_does_not_compact(self):
        session = self.seed(6)
        boundary = harness.request_size(self.payload(session))
        for threshold in (boundary, boundary + 1):
            with self.subTest(threshold=threshold), patch.object(main, "AUTOCOMPACT_SOFT_BYTES", threshold):
                same = main.auto_compact_before_question(session, self.store, "新问题", CONFIG, True)
                self.assertIs(same, session)
        self.transport.assert_not_called()

    def test_just_over_threshold_compacts_only_eligible_whole_turn(self):
        session = self.seed(6, tool_index=0)
        original = copy.deepcopy(self.turns(session))
        self.script = [self.response(self.content([self.fact(original[0])])), reply("继续回答")]
        with patch.object(main, "AUTOCOMPACT_SOFT_BYTES", harness.request_size(self.payload(session)) - 1):
            status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        batch = json.loads(self.requests[0]["messages"][1]["content"])["batch"]
        self.assertEqual(batch, [{"turn_id": original[0]["id"], "messages": original[0]["messages"]}])
        self.assertEqual(len(batch[0]["messages"]), 4)
        self.assertEqual(self.turns(self.read(session["id"]))[:6], original)
        self.assertIn("自动整理已保存", output)
        self.clock.assert_not_called()
        self.weather_transport.assert_not_called()

    def test_success_rebuilds_question_context_and_records_separate_usage(self):
        session = self.large_history()
        original = copy.deepcopy(self.turns(session))
        self.script = [self.response(self.content([self.fact(original[0])])), metered(reply("你喜欢37"), 50, 10)]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "喜欢什么", "/runs", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 2)
        batch = json.loads(self.requests[0]["messages"][1]["content"])["batch"]
        self.assertEqual([t["turn_id"] for t in batch], [t["id"] for t in original[:5]])
        self.assertNotIn("喜欢什么", [m["content"] for t in batch for m in t["messages"]])
        sent = self.requests[1]["messages"]
        self.assertIn("37", sent[1]["content"])
        self.assertEqual(sent[2:-1], [m for t in original[5:] for m in t["messages"]])
        self.assertEqual(sent[-1], {"role": "user", "content": "喜欢什么"})
        saved = self.read(session["id"])
        self.assertEqual(saved["version"], 4)
        self.assertEqual(self.turns(saved)[:10], original)
        self.assertEqual([t["kind"] for t in self.turns(saved)[-2:]], ["compaction", "chat"])
        self.assertEqual([t["runs"][0]["usage"]["total_tokens"] for t in self.turns(saved)[-2:]], [120, 60])
        self.assertEqual(self.store.load(session["id"]), saved)
        self.assertIn("第1/2批", output)
        self.assertNotIn("第2/2批", output)

    def test_two_batch_cap_does_not_drop_remaining_history(self):
        session = self.large_history(25)
        fact = self.fact(self.turns(session)[0])
        self.script = [self.response(self.content([fact])), self.response(self.content([fact])), reply("正常回答")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 3)
        self.assertIn("两批上限", output)
        saved = self.read(session["id"])
        memory = saved["threads"][saved["active_thread"]]["memory"]
        self.assertEqual(memory["revision"], 2)
        self.assertEqual(len(memory["covered_turn_ids"]), 10)
        self.assertEqual(self.requests[-1]["messages"][2:-1],
                         [m for t in self.turns(session)[10:] for m in t["messages"]])

    def test_failure_preserves_original_context_and_stops_retrying_this_question(self):
        for error in (TimeoutError(FAKE_KEY), metered(reply("{invalid"))):
            with self.subTest(error_type=type(error).__name__):
                self.requests.clear()
                session = self.large_history()
                original = copy.deepcopy(self.turns(session))
                self.script = [error, reply("仍可回答")]
                status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
                self.assertEqual(status, 0)
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(self.requests[-1], self.payload(session))
                saved = self.read(session["id"])
                self.assertEqual(saved["threads"][saved["active_thread"]]["memory"], harness.empty_memory())
                self.assertEqual(self.turns(saved)[:10], original)
                self.assertEqual([t["status"] for t in self.turns(saved)[-2:]], ["failed", "completed"])
                self.assertIn("自动整理失败", output)
                self.assertIn("本次不再自动重试", output)

    def test_second_batch_failure_keeps_first_successful_memory(self):
        session = self.large_history(16)
        fact = self.fact(self.turns(session)[0])
        self.script = [self.response(self.content([fact])), TimeoutError(FAKE_KEY), reply("使用第一批记忆继续")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"]["revision"], 1)
        self.assertIn("37", self.requests[-1]["messages"][1]["content"])
        self.assertEqual(self.requests[-1]["messages"][2:-1], [m for t in self.turns(session)[5:] for m in t["messages"]])
        self.assertEqual([t["status"] for t in self.turns(saved)[-3:]], ["completed", "failed", "completed"])

    def test_oversize_question_remains_blocked_after_two_batches(self):
        session = self.large_history(25)
        fact = self.fact(self.turns(session)[0])
        self.script = [self.response(self.content([fact])), self.response(self.content([fact]))]
        question = "长" * 45000
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", question, "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all("response_format" in request for request in self.requests))
        self.assertIn("128 KiB", output)
        saved = self.read(session["id"])
        self.assertEqual(self.turns(saved)[-1]["status"], "failed")
        self.assertEqual(self.turns(saved)[-1]["runs"][0]["usage"]["model_calls"], 0)
        self.assertEqual(len(harness.completed_turns(saved["threads"][saved["active_thread"]])), 25)

    def test_over_hard_limit_history_can_be_reduced_before_chat(self):
        session = self.large_history(10, answer_chars=4500)
        self.assertGreater(harness.request_size(self.payload(session)), harness.MAX_REQUEST_BYTES)
        self.script = [self.response(self.content()), reply("压缩后可发送")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 2)
        self.assertLess(harness.request_size(self.requests[-1]), harness.MAX_REQUEST_BYTES)
        self.assertIn("压缩后可发送", output)

    def test_only_five_recent_turns_are_never_compacted_even_if_huge(self):
        session = self.large_history(5, answer_chars=10000)
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("没有可整理的旧问答", output)
        self.transport.assert_not_called()
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"], harness.empty_memory())
        self.assertEqual(self.turns(saved)[:-1], self.turns(session))

    def test_oversize_compaction_is_not_sent_and_chat_hard_limit_still_applies(self):
        session = self.large_history(10, answer_chars=10000)
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("自动整理失败", output)
        self.transport.assert_not_called()
        saved = self.read(session["id"])
        self.assertEqual([t["status"] for t in self.turns(saved)[-2:]], ["failed", "failed"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"], harness.empty_memory())

    def test_switching_threads_only_compacts_active_thread(self):
        session = self.large_history()
        a = session["active_thread"]
        a_before = copy.deepcopy(session["threads"][a])
        session = harness.new_thread(session, self.store)
        b = session["active_thread"]
        for i in range(10):
            self.add_chat(session, "我喜欢92" if i == 0 else f"B问题{i}", "资料" * 750)
        fact = self.fact(self.turns(session)[0], value="92")
        self.script = [self.response(self.content([fact])), reply("92")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "B的新问题", "exit"])
        self.assertEqual(status, 0)
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][a], a_before)
        self.assertEqual(saved["threads"][b]["memory"]["revision"], 1)
        batch = json.loads(self.requests[0]["messages"][1]["content"])["batch"]
        self.assertEqual([t["turn_id"] for t in batch], [t["id"] for t in self.turns(session)[:5]])
        self.assertFalse({t["turn_id"] for t in batch} & {t["id"] for t in a_before["turns"]})
        self.assertNotIn("我喜欢的数字是37，请记住。", [m["content"] for m in self.requests[-1]["messages"]])

    def test_off_after_on_disables_future_automatic_requests(self):
        session = self.large_history()
        self.script = [reply("不自动整理")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "/autocompact off", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(self.requests, [self.payload(session)])
        self.assertNotIn("自动整理已保存", output)

    def test_five_large_recent_turns_below_hard_limit_still_allow_question(self):
        session = self.large_history(5, answer_chars=3000)
        self.script = [reply("保留原文回答")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(self.requests, [self.payload(session)])
        self.assertIn("没有可整理的旧问答", output)

    def test_hard_limit_after_tool_blocks_followup_without_midturn_compaction(self):
        session = self.large_history()
        self.script = [self.response(self.content()), reply(None, [time_call()])]
        with patch.object(main, "execute", return_value={"ok": True, "result": {"large": "资" * 45000}}):
            status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "查询时间", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(["response_format" in p for p in self.requests], [True, False])
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"]["revision"], 1)
        self.assertEqual(self.turns(saved)[-1]["status"], "failed")
        self.assertEqual(self.turns(saved)[-1]["messages"], [])
        self.assertEqual([s["step_type"] for s in self.turns(saved)[-1]["runs"][0]["steps"]], ["llm", "tool"])
        self.assertIn("128 KiB", output)

    def test_policy_stays_on_across_new_load_thread_and_use_but_resets_next_launch(self):
        session = self.seed(1)
        a = session["active_thread"]
        status, output = self.run_cli(["/autocompact on", "/new", "/load " + session["id"], "/thread",
                                       "/use " + a, "/autocompact", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(output.count("自动整理：开启"), 2)
        self.assertNotIn("auto_enabled", self.store.path(session["id"]).read_text(encoding="utf-8"))
        status, output = self.run_cli(["/load " + session["id"], "/autocompact", "exit"])
        self.assertEqual(status, 0)
        self.assertIn("自动整理：关闭", output)
        self.transport.assert_not_called()

    def test_tool_followup_never_compacts_in_middle_of_turn(self):
        session = self.large_history()
        self.script = [self.response(self.content()), reply(None, [time_call()]), reply("工具已返回时间")]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "查询本机时间", "exit"])
        self.assertEqual(status, 0)
        self.assertEqual(["response_format" in p for p in self.requests], [True, False, False])
        self.assertEqual(self.requests[-1]["messages"][-1]["role"], "tool")
        saved = self.read(session["id"])
        self.assertEqual(len(self.turns(saved)[-1]["messages"]), 4)
        self.clock.assert_called_once()

    def test_auto_storage_failure_stops_before_question_request(self):
        session = self.large_history()
        before = self.store.path(session["id"]).read_bytes()
        self.script = [self.response(self.content())]
        original_save = harness.Store.save

        def failing_save(store, candidate):
            turns = candidate["threads"][candidate["active_thread"]]["turns"]
            if candidate["id"] == session["id"] and turns[-1]["kind"] == "compaction" and turns[-1]["status"] == "completed":
                raise harness.StorageError("写盘失败，已停止。")
            return original_save(store, candidate)

        with patch.object(harness.Store, "save", new=failing_save):
            status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题"])
        self.assertEqual(status, 1)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("自动整理已保存", output)
        self.assertNotIn("模型回答：", output)
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"], harness.empty_memory())
        self.assertEqual(self.turns(saved)[-1]["status"], "running")
        self.assertEqual(self.turns(saved)[:-1], self.turns(session))
        self.assertNotEqual(self.store.path(session["id"]).read_bytes(), before)
        loaded = self.store.load(session["id"])
        self.assertEqual(self.turns(loaded)[-1]["status"], "interrupted")

    def test_cancel_on_second_batch_keeps_first_success_and_does_not_ask_question(self):
        session = self.large_history(16)
        self.script = [self.response(self.content()), KeyboardInterrupt()]
        status, output = self.run_cli(["/load " + session["id"], "/autocompact on", "新问题"])
        self.assertEqual(status, 130)
        self.assertEqual(len(self.requests), 2)
        saved = self.read(session["id"])
        self.assertEqual(saved["threads"][saved["active_thread"]]["memory"]["revision"], 1)
        self.assertEqual([t["status"] for t in self.turns(saved)[-2:]], ["completed", "interrupted"])
        self.assertIn("当前问题未加入历史", output)

    def test_v3_and_v4_sessions_load_without_policy_or_model_calls(self):
        for version in (3, 4):
            with self.subTest(version=version):
                session = self.large_history()
                if version == 3:
                    self.store.save(as_legacy(session, 3))
                status, output = self.run_cli(["/load " + session["id"], "/context", "/autocompact", "exit"])
                self.assertEqual(status, 0)
                self.assertIn("自动整理：关闭", output)
                loaded = self.store.load(session["id"])
                self.assertEqual(loaded["version"], 4)
                self.assertEqual(loaded["threads"][loaded["active_thread"]]["memory"], harness.empty_memory())
        self.transport.assert_not_called()


class ConfigurationTests(OfflineCase):
    def test_missing_config_stops_without_input_or_request(self):
        self.environment.clear()
        output = io.StringIO()
        with patch("builtins.input") as user_input, contextlib.redirect_stdout(output):
            status = main.main()
        self.assertEqual(status, 1)
        self.assertIn("缺少配置", output.getvalue())
        self.transport.assert_not_called()
        user_input.assert_not_called()

    def test_virtual_env_file_and_terminal_priority(self):
        self.env_file.read_text.return_value = (
            '# 注释\nDEEPSEEK_API_KEY="offline-dummy-key"\n'
            'DEEPSEEK_MODEL=file-model\nAGENT_SYSTEM_PROMPT="说明 a=b"\n'
        )
        self.environment.clear()
        self.assertEqual(main.load_config()["AGENT_SYSTEM_PROMPT"], "说明 a=b")
        self.environment["DEEPSEEK_MODEL"] = "terminal-model"
        self.assertEqual(main.load_config()["DEEPSEEK_MODEL"], "terminal-model")
        self.environment["DEEPSEEK_MODEL"] = " "
        self.assertEqual(main.load_config()["DEEPSEEK_MODEL"], "file-model")

    def test_missing_env_file_uses_terminal(self):
        self.env_file.read_text.side_effect = FileNotFoundError()
        self.assertEqual(main.load_config(), CONFIG)

    def test_bad_env_errors_hide_values(self):
        for text in (FAKE_KEY, "WRONG=" + FAKE_KEY,
                     'DEEPSEEK_API_KEY="' + FAKE_KEY,
                     'DEEPSEEK_API_KEY=x\nDEEPSEEK_API_KEY=y'):
            with self.subTest(text=text):
                self.env_file.read_text.return_value = text
                status, output = self.run_cli([])
                self.assertEqual(status, 1)
                self.assertIn("配置读取失败", output)
        self.transport.assert_not_called()

    def test_unreadable_env_has_safe_error(self):
        self.env_file.read_text.side_effect = PermissionError(FAKE_KEY)
        status, output = self.run_cli([])
        self.assertEqual(status, 1)
        self.assertIn("无法读取 .env", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
