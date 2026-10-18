"""工具：有限算术、本机时间，以及 Open-Meteo 当前天气查询。"""

import ast
import json
import math
import operator
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MAX_EXPRESSION_LENGTH = 200
MAX_ARGUMENTS_LENGTH = 2_000
MAX_AST_NODES = 80
MAX_DEPTH = 16
MAX_EXPONENT = 12
MAX_ABSOLUTE_VALUE = 10**12
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 10
MAX_WEATHER_RESPONSE_BYTES = 512 * 1024
MAX_CITY_LENGTH = 80
MAX_LOCATION_CANDIDATES = 5

WEATHER_DESCRIPTIONS = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "中等毛毛雨", 55: "浓密毛毛雨",
    56: "轻微冻毛毛雨", 57: "强冻毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    66: "轻微冻雨", 67: "强冻雨", 71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒", 80: "小阵雨", 81: "中等阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴小冰雹",
    97: "强雷暴", 99: "雷暴伴大冰雹",
}

SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算有限的算术表达式，使用 + - * / // % ** 和括号。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "算术表达式，例如 23*47。",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "获取运行此程序的电脑当前本地时间及 UTC 偏移。不支持指定城市或时区。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定地点的当前温度和天气状况。多个候选地点时须向用户确认。不支持未来或历史天气。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称；必要时使用 城市, 国家或省份，例如 Paris, France。",
                    "minLength": 1,
                    "maxLength": MAX_CITY_LENGTH,
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]


class ToolError(ValueError):
    """只包含程序自己生成的、可安全显示的工具错误。"""


def _local_now():
    """正常运行时读取系统时间；离线测试可替换这个入口。"""
    return datetime.now().astimezone()


def get_time():
    now = _local_now()
    if now.utcoffset() is None:
        raise ToolError("本机时间缺少有效的 UTC 偏移。")
    return {"local_time": now.isoformat(timespec="seconds")}


def _fetch_json(url, parameters):
    """仅请求两个固定天气服务地址；不携带模型密钥，不自动重试。"""
    services = {GEOCODING_URL: "地点查询服务", WEATHER_URL: "天气服务"}
    if url not in services:
        raise ToolError("天气服务地址不受支持。")
    service = services[url]
    request = Request(url + "?" + urlencode(parameters),
                      headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=WEATHER_TIMEOUT) as response:
            raw = response.read(MAX_WEATHER_RESPONSE_BYTES + 1)
    except HTTPError as error:
        hint = "请求受到限流，请稍后手动重试。" if error.code == 429 else "暂时无法完成请求。"
        raise ToolError(f"{service}请求失败（HTTP {error.code}）：{hint}") from None
    except (TimeoutError, URLError, OSError):
        raise ToolError(f"{service}超时或连接失败，请检查网络或稍后手动重试。") from None
    if len(raw) > MAX_WEATHER_RESPONSE_BYTES:
        raise ToolError(f"{service}响应过大。")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ToolError(f"{service}未返回有效或可处理的 UTF-8 JSON。") from None
    if not isinstance(data, dict):
        raise ToolError(f"{service}响应必须是 JSON 对象。")
    if data.get("error") is True:
        raise ToolError(f"{service}返回错误，未得到有效数据。")
    return data


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _place_label(place):
    if not isinstance(place, dict):
        raise ToolError("地点候选数据格式不正确。")
    name = place.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise ToolError("地点候选缺少有效名称。")
    label = {"name": name.strip()}
    for field in ("admin1", "country"):
        value = place.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > 200):
            raise ToolError("地点候选的省份或国家格式不正确。")
        label[field] = value
    return label


def get_weather(city):
    if not isinstance(city, str) or not city.strip():
        raise ToolError("city 必须是非空字符串。")
    if len(city) > MAX_CITY_LENGTH:
        raise ToolError("city 不能超过 80 字符。")
    data = _fetch_json(GEOCODING_URL, {
        "name": city.strip(), "count": MAX_LOCATION_CANDIDATES, "language": "zh",
    })
    # 地点服务在无匹配结果时可能省略 results。
    choices = data.get("results", [])
    if not isinstance(choices, list) or len(choices) > MAX_LOCATION_CANDIDATES:
        raise ToolError("地点候选列表格式或数量不正确。")
    if not choices:
        raise ToolError("未找到地点，请尝试更准确的名称，或使用 城市, 国家或省份。")
    labels = [_place_label(place) for place in choices]
    if len(choices) > 1:
        return {"needs_clarification": True, "candidates": labels}

    place = choices[0]
    latitude, longitude = place.get("latitude"), place.get("longitude")
    if (not _finite_number(latitude) or not -90 <= latitude <= 90
            or not _finite_number(longitude) or not -180 <= longitude <= 180):
        raise ToolError("地点缺少有效的经纬度。")
    data = _fetch_json(WEATHER_URL, {
        "latitude": latitude, "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius", "timezone": "auto",
    })
    current, units = data.get("current"), data.get("current_units")
    if not isinstance(current, dict) or not isinstance(units, dict):
        raise ToolError("天气响应缺少有效的 current 或 current_units。")
    time_text = current.get("time")
    if not isinstance(time_text, str) or "T" not in time_text or len(time_text) > 64:
        raise ToolError("天气响应缺少有效的数据时间。")
    try:
        datetime.fromisoformat(time_text)
    except ValueError:
        raise ToolError("天气数据时间格式不正确。") from None
    temperature, code = current.get("temperature_2m"), current.get("weather_code")
    if not _finite_number(temperature):
        raise ToolError("天气响应缺少有效的温度数值。")
    if type(code) is not int or not 0 <= code <= 999:
        raise ToolError("天气响应缺少有效的天气编码。")
    if units.get("temperature_2m") != "°C":
        raise ToolError("天气响应缺少摄氏度单位，未将数值当作摄氏温度。")
    zone, offset = data.get("timezone"), data.get("utc_offset_seconds")
    if (not isinstance(zone, str) or not zone.strip() or len(zone) > 128
            or type(offset) is not int or not -86400 < offset < 86400):
        raise ToolError("天气响应缺少有效的时区或 UTC 偏移。")
    return {
        "needs_clarification": False,
        "location": {**labels[0], "latitude": latitude, "longitude": longitude},
        "current": {
            "time": time_text, "temperature_2m": temperature, "weather_code": code,
            "weather_description": WEATHER_DESCRIPTIONS.get(code, f"未知天气编码（{code}）"),
        },
        "units": {"temperature_2m": units["temperature_2m"]},
        "timezone": zone, "utc_offset_seconds": offset, "source": "Open-Meteo",
    }


def calculate(expression):
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression 必须是非空字符串。")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError("表达式不能超过 200 字符。")
    try:
        # mode='eval' 只指定表达式语法；这里没有调用 eval()。
        root = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError):
        raise ToolError("表达式语法不正确。") from None
    if sum(1 for _ in ast.walk(root)) > MAX_AST_NODES:
        raise ToolError("表达式节点过多。")
    operations = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def bounded(value):
        # bool 是 int 的子类，所以使用精确类型检查来排除布尔值。
        if type(value) not in (int, float):
            raise ToolError("只允许整数和有限浮点数，不支持布尔值或复数。")
        if abs(value) > MAX_ABSOLUTE_VALUE or not math.isfinite(value):
            raise ToolError("数值必须有限，且绝对值不能超过 10^12。")
        return value

    def visit(node, depth=0):
        if depth > MAX_DEPTH:
            raise ToolError("表达式嵌套过深。")
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand, depth + 1)
            return bounded(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError("指数绝对值不能超过 12。")
            try:
                value = operations[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ToolError("不能除以零。") from None
            except ArithmeticError:
                raise ToolError("计算失败或数值超出范围。") from None
            return bounded(value)
        raise ToolError("只允许数字和算术运算，不允许变量、函数调用或属性访问。")

    return {"value": visit(root.body)}


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ToolError("工具参数包含重复字段。")
        result[name] = value
    return result


def execute(name, arguments, *, overrides=None):
    """校验调用后执行已注册函数；错误作为数据返回给模型。"""
    try:
        functions = {"calculate": calculate, "get_time": get_time, "get_weather": get_weather}
        required_arguments = {"calculate": {"expression"}, "get_time": set(), "get_weather": {"city"}}
        if not isinstance(name, str) or name not in functions:
            raise ToolError("工具未注册，本阶段只支持 calculate、get_time 和 get_weather。")
        if overrides is not None:
            if (not isinstance(overrides, dict) or not set(overrides) <= set(functions)
                    or not all(callable(fn) for fn in overrides.values())):
                raise ToolError("工具实现替换配置不正确。")
            # 仅供受信任的本地调用方注入；模型参数没有此配置入口。
            functions.update(overrides)
        if not isinstance(arguments, str) or len(arguments) > MAX_ARGUMENTS_LENGTH:
            raise ToolError("工具参数必须是 JSON 字符串，且不能超过 2000 字符。")
        try:
            args = json.loads(arguments, object_pairs_hook=unique_object)
        except (json.JSONDecodeError, RecursionError):
            raise ToolError("工具参数不是有效或可处理的 JSON。") from None
        if not isinstance(args, dict):
            raise ToolError("工具参数必须是 JSON 对象。")
        if set(args) != required_arguments[name]:
            if name == "get_time":
                raise ToolError("get_time 不接受任何参数。")
            if name == "get_weather":
                raise ToolError("get_weather 的参数必须且只能包含 city。")
            raise ToolError("calculate 的参数必须且只能包含 expression。")
        return {"ok": True, "result": functions[name](**args)}
    except ToolError as error:
        return {"ok": False, "error_type": "ToolError", "error": str(error)}
    except Exception:
        return {"ok": False, "error_type": "ToolError", "error": "工具未能完成请求。"}
