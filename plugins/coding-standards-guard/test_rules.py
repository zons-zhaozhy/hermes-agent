"""coding-standards-guard 规则测试。

用 AST 精准检测，正反用例覆盖全部规则。
"""
import importlib.util
import sys
import os

# 手动加载插件（模拟 Hermes 插件加载机制）
_plugin_dir = os.path.join(os.path.dirname(__file__))
_spec = importlib.util.spec_from_file_location(
    "coding_standards_guard",
    os.path.join(_plugin_dir, "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

check_code = _mod._run_all_checks

PASS_COUNT = 0
FAIL_COUNT = 0


def test(name: str, code: str, should_detect_rule: str | None):
    """Test a code snippet.

    Args:
        name: test case description
        code: Python source code
        should_detect_rule: expected rule_id to detect, or None for clean code
    """
    global PASS_COUNT, FAIL_COUNT
    violations = check_code(code)
    detected_rules = {v.rule_id for v in violations}

    if should_detect_rule is None:
        if not violations:
            PASS_COUNT += 1
        else:
            FAIL_COUNT += 1
            for v in violations:
                print(f"  FAIL: '{name}' — 预期无违规，但检测到: {v.rule_id}")
                print(f"    {v.line} [{v.rule_id}] {v.message}")
    else:
        if should_detect_rule in detected_rules:
            PASS_COUNT += 1
        else:
            FAIL_COUNT += 1
            print(f"  FAIL: '{name}' — 预期检测到 {should_detect_rule}，实际: {detected_rules or '无'}")


# ═══════════════════════════════════════════════════════════════════════
# 吞异常系列
# ═══════════════════════════════════════════════════════════════════════
print("R001-R005: except body 只有 pass")
test("R001 except Exception: pass",
     "try:\n    pass\nexcept Exception:\n    pass", "R001")
test("R002 bare except: pass",
     "try:\n    pass\nexcept:\n    pass", "R002")
test("R003 except Exception as e: pass",
     "try:\n    pass\nexcept Exception as e:\n    pass", "R003")
test("R005 except ValueError: pass",
     "try:\n    pass\nexcept ValueError:\n    pass", "R005")
test("允许: except Exception with logger",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    pass\nexcept Exception as e:\n    logger.warning(e)", None)
test("允许: except with raise",
     "try:\n    pass\nexcept Exception as e:\n    raise", None)
test("允许: except ValueError return False（正常输入校验）",
     "try:\n    x = int(s)\nexcept ValueError:\n    return False", None)
test("允许: except TypeError return False",
     "try:\n    x = int(s)\nexcept TypeError:\n    return False", None)

print("R008: 静默降级 — 宽异常 + return 常量")
test("R008 except Exception return None",
     "try:\n    pass\nexcept Exception:\n    return None", "R008")
test("R008 except Exception return []",
     "try:\n    pass\nexcept Exception:\n    return []", "R008")
test("R008 bare except return {}",
     "try:\n    pass\nexcept:\n    return {}", "R008")
test("允许: except ValueError return False（R008 不报）",
     "try:\n    pass\nexcept ValueError:\n    return False", None)

print("R013: 静默吞异常 — 宽异常 + return + 无 logger")
test("R013 except Exception return dict",
     "try:\n    pass\nexcept Exception:\n    return {'error': 'oops'}", "R013")
test("R013 except Exception return + print",
     "try:\n    pass\nexcept Exception as e:\n    print(e)\n    return None", "R013")
test("允许: except Exception return + logger.warning",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    pass\nexcept Exception as e:\n    logger.warning(e)\n    return None", None)
test("允许: except ValueError return False（R013 不报）",
     "try:\n    pass\nexcept ValueError:\n    return False", None)
test("允许: except Exception return + raise",
     "try:\n    pass\nexcept Exception as e:\n    return e\n    raise", None)

print("R015: except 块只有 debug/info 无 warning")
test("R015 only logger.debug",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    pass\nexcept Exception as e:\n    logger.debug(e)", "R015")
test("R015 only logger.info",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    pass\nexcept Exception as e:\n    logger.info(e)", "R015")
test("允许: logger.debug + logger.warning",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    pass\nexcept Exception as e:\n    logger.debug('details')\n    logger.warning(e)", None)
test("允许: only logger.warning",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    pass\nexcept Exception as e:\n    logger.warning(e)", None)

# ═══════════════════════════════════════════════════════════════════════
# R020: 变换异常信息 — except 内 raise 新异常不带 from e
# ═══════════════════════════════════════════════════════════════════════
print("R020: 变换异常信息 — raise 新异常不带 from e")
test("R020 raise 新异常无 from e",
     'try:\n    x = 1\nexcept Exception as e:\n    raise ValueError(f"bad: {e}")\n', "R020")
test("R020 raise 新异常 from e — 显式链放行",
     'try:\n    x = 1\nexcept Exception as e:\n    raise ValueError(f"bad: {e}") from e\n', None)
test("R020 裸 raise — 完整透传放行",
     'try:\n    x = 1\nexcept Exception as e:\n    raise\n', None)
test("R020 raise e — re-raise 本体放行",
     'try:\n    x = 1\nexcept Exception as e:\n    raise e\n', None)
test("允许: from None — 显式压制=主动声明(0901改判,原warning撤)",
     'try:\\\\n    x = 1\\\\nexcept Exception as e:\\\\n    raise RuntimeError("x") from None\\\\n', None)
test("R020 豁免: # raise-ok — except内业务校验分支raise",
     'try:\\\\n    x = 1\\\\nexcept Exception as e:\\\\n    if not check():\\\\n        raise ValueError("bad")  # raise-ok 查重失败与原异常无因果\\\\n', None)
test("R020 raise 局部变量新异常 — 同样断链",
     'try:\n    x = 1\nexcept Exception as e:\n    exc = RuntimeError("x")\n    raise exc\n', "R020")
test("R020 except 外 raise — 放行",
     'def f():\n    raise ValueError("no except here")\n', None)


# ═══════════════════════════════════════════════════════════════════════
# 默认值兜底系列
# ═══════════════════════════════════════════════════════════════════════
print("R009: 默认值兜底")
test("R009 os.environ.get with default",
     "DB_HOST = os.environ.get('DB_HOST', 'localhost')", "R009")
test("允许: os.environ.get no default",
     "DB_HOST = os.environ.get('DB_HOST')", None)
test("允许: os.environ.get empty default",
     "DB_HOST = os.environ.get('DB_HOST', '')", None)

print("R016: getattr 静默降级")
test("R016 getattr(config, key, default)",
     "timeout = getattr(config, 'timeout', 30)", "R016")
test("R016 getattr(settings, key, default)",
     "debug = getattr(settings, 'debug', False)", "R016")
test("允许: getattr(obj, 'attr', None) — Python 惯用法",
     "agent = getattr(cli, 'agent', None)", None)
test("允许: getattr(self, '_attr', False) — 实例属性",
     "app = getattr(self, '_app', None)", None)

print("R018: 别名兼容")
test("R018 config.get old or new",
     "db = config.get('old_db') or config.get('new_db')", "R018")
test("允许: config.get single",
     "db = config.get('db_host')", None)

# ═══════════════════════════════════════════════════════════════════════
# 硬编码系列
# ═══════════════════════════════════════════════════════════════════════
print("R007: 硬编码密码")
test("R007 password = 'xxx'",
     "DB_PASSWORD = 'my_secret_123'", "R007")
test("R007 api_key = 'xxx'",
     "api_key = 'sk-1234567890abcdef'", "R007")
test("允许: password from env",
     "DB_PASSWORD = os.environ.get('DB_PASSWORD')", None)
test("允许: password = None",
     "DB_PASSWORD = None", None)

print("R010: 硬编码 IP")
test("R010 192.168.x.x",
     "HOST = '192.168.1.100'", "R010")
test("R010 localhost:port",
     "URL = 'localhost:5432'", "R010")
test("允许: 127.0.0.1（不是私有网段）",
     "LOCALHOST = '127.0.0.1'", None)

print("R011: 硬编码 DB URL")
test("R011 postgresql://",
     "DATABASE_URL = 'postgresql://user:pass@host/db'", "R011")
test("允许: from env",
     "DATABASE_URL = os.environ.get('DATABASE_URL')", None)

print("R012: 硬编码部署路径")
test("R012 /opt/",
     "DEPLOY_DIR = '/opt/ontox/deploy'", "R012")
test("允许: 相对路径",
     "DATA_DIR = './data'", None)

print("硬编码系列 — kwarg/AnnAssign/dict 形态（_iter_name_value_pairs 扩面）")
test("R007 kwarg password",
     "connect(password='my_secret_123')", "R007")
test("R007 dict db_password",
     "CFG = {'db_password': 'abc123'}", "R007")
test("R010 kwarg host",
     "connect(host='192.168.1.5')", "R010")
test("R010 AnnAssign localhost:5432",
     "HOST: str = 'localhost:5432'", "R010")
test("R011 dict url",
     "CFG = {'url': 'postgresql://u:p@h/d'}", "R011")
test("R012 kwarg deploy_dir",
     "f(deploy_dir='/opt/ontox/x')", "R012")
test("允许: kwarg 非字面量",
     "connect(password=os.environ['PW'])", None)

print("R021: 正则使用")
test("R021 re.sub",
     "import re\nx = re.sub('a', 'b', s)", "R021")
test("R021 re.compile 链式",
     "m = re.compile(r'\\d+').match(s)", "R021")
test("R021 豁免 # re-ok",
     "import re  # re-ok 确需提取数字\nx = re.match(r'\\d+', s)  # re-ok", None)
test("允许: str.replace",
     "x = s.replace('a', 'b')", None)

# ═══════════════════════════════════════════════════════════════════════
# 安全 + 代码质量
# ═══════════════════════════════════════════════════════════════════════
print("R014: 裸 eval()")
test("R014 eval()",
     "result = eval(user_input)", "R014")
test("允许: ast.literal_eval",
     "import ast\nresult = ast.literal_eval(user_input)", None)

print("R017: str(val or \"\") 仅数值语义")
test("R017 str(row_val or \"\")",
     "row_str = str(row_value or \"\")", "R017")
test("R017 str(count or \"\")",
     "count_str = str(count or \"\")", "R017")
test("允许: str(name or \"\") — name 不是数值",
     "name_str = str(name or \"\")", None)
test("允许: str(title or \"\")",
     "title_str = str(title or \"\")", None)

print("R019: 函数内 sys.exit() 非入口函数")
test("R019 sys.exit in helper",
     "def helper():\n    import sys\n    sys.exit(1)", "R019")
test("允许: sys.exit in main()",
     "def main():\n    import sys\n    sys.exit(0)", None)
test("允许: sys.exit in cli()",
     "def cli():\n    import sys\n    sys.exit(1)", None)

print("R006: import *")
test("R006 from X import *",
     "from os.path import *", "R006")
test("允许: from X import specific",
     "from os.path import join, exists", None)

# ═══════════════════════════════════════════════════════════════════════
# 复合场景
# ═══════════════════════════════════════════════════════════════════════
print("复合场景")
test("合法: ValueError+return False（不触发 R008/R013）",
     "def parse_int(s):\n    try:\n        return int(s)\n    except ValueError:\n        return 0", None)
test("合法: OSError+logger.warning+return",
     "import logging\nlogger = logging.getLogger(__name__)\ntry:\n    f = open(path)\nexcept OSError as e:\n    logger.warning('file error')\n    return None", None)
test("违规: Exception+pass",
     "try:\n    do_something()\nexcept Exception:\n    pass", "R001")
test("违规: except+return None 无日志",
     "try:\n    do_something()\nexcept Exception:\n    return None", "R008")

# 复合场景之外的独立规则用例须插在结果打印之前
_R022_CASES = [
    ("R022 违规: print 消息截断",
     'def f(msg):\n    print(f"派发: {msg[:120]}")', "R022"),
    ("R022 违规: logger 参数截断",
     'import logging\nlogger = logging.getLogger(__name__)\ndef f(e):\n    logger.warning("失败: %s", str(e)[-300:])', "R022"),
    ("R022 违规: raise 消息截断",
     'def f(e):\n    raise RuntimeError(f"err {str(e)[:300]}") from e', "R022"),
    ("R022 合法: 非诊断切片(data[:8000] 赋值)",
     'def f(data):\n    body = data[:8000]\n    return body', None),
    ("R022 合法: 截断+trunc-ok 豁免",
     'def f(msg):\n    print(f"{msg[:100]}")  # trunc-ok 回灌LLM功能性上限', None),
]
print("R022: 诊断输出截断")
for _name, _code, _expect in _R022_CASES:
    test(_name, _code, _expect)

# R023 期望值独立推导: 「有而不取」=命令字面量含 logs/journalctl 且无时间戳参数
_R023_CASES = [
    ("R023 违规: docker logs 无 --timestamps(doctor 实际形态)",
     'def f(name, since):\n    subprocess.run(["docker", "logs", "--since", since, name])', "R023"),
    ("R023 违规: journalctl 无时间输出格式",
     'def f(u):\n    subprocess.run(["journalctl", "-u", u, "--no-pager"])', "R023"),
    ("R023 合法: 带 --timestamps",
     'def f(name):\n    subprocess.run(["docker", "logs", "--timestamps", name])', None),
    ("R023 合法: ts-ok 豁免(纯计数场景)",
     'def f(name):\n    subprocess.run(["docker", "logs", name])  # ts-ok 仅统计行数,时间无关', None),
    ("R023 合法: 非采集调用(变量名 logs 作他用)",
     'def f(logs):\n    print(len(logs))', None),
]
print("R023: 日志采集缺时间戳")
for _name, _code, _expect in _R023_CASES:
    test(_name, _code, _expect)

print("\n" + "=" * 60)
print(f"结果: {PASS_COUNT}/{PASS_COUNT + FAIL_COUNT} 通过, {FAIL_COUNT}/{PASS_COUNT + FAIL_COUNT} 失败")
if FAIL_COUNT == 0:
    print("全部通过！")
else:
    print(f"有 {FAIL_COUNT} 个测试失败！")
