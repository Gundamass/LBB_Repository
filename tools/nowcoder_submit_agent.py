#!/usr/bin/env python3
"""Semi-automatic Nowcoder competition submitter.

This script intentionally does not store account passwords. The recommended
workflow is:

1. Run with --login once, sign in in the opened browser, then press Enter.
2. Run without --login to submit ZIP files from a queue.

The page structure of competition websites can change, so selectors are
configurable from the command line. The default mode uses conservative
heuristics for common Chinese upload/submit buttons.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_URL = "https://competition.nowcoder.com/exam/226/419"
DEFAULT_WORK_DIR = Path("outputs/.nowcoder_submitter")
DEFAULT_STATE = DEFAULT_WORK_DIR / "state.json"
DEFAULT_QUEUE = Path("submit_queue_nowcoder.txt")
DEFAULT_TASK_NAME = "少样本条件下电子产品外观缺陷检测"

UPLOAD_BUTTON_RE = re.compile(
    r"(上传|提交|选择文件|提交结果|提交文件|重新提交|我要参赛|开始答题|Upload|Submit|Choose)",
    re.I,
)
SUBMIT_BUTTON_RE = re.compile(
    r"(确认|确定|提交|上传|提交结果|提交文件|Submit|Upload|OK)",
    re.I,
)
SCORE_RE = re.compile(r"\b0\.\d{4,6}\b")
LOGIN_CHALLENGE_RE = re.compile(r"(安全验证|滑块|captcha|人机验证|请完成验证|拖动滑块|图形验证码)", re.I)
LOGIN_ERROR_RE = re.compile(r"(请输入正确的手机号码|账号或密码错误|密码错误|账号不存在|登录失败|请输入密码|请输入邮箱/手机号码)")


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def mask_account(account: str) -> str:
    if len(account) <= 4:
        return "*" * len(account)
    return f"{account[:2]}{'*' * max(2, len(account) - 4)}{account[-2:]} (len={len(account)})"


def read_queue(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Queue file not found: {path}")
    items: List[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"ZIP in queue does not exist: {p}")
        if p.suffix.lower() != ".zip":
            raise ValueError(f"Queue item is not a .zip file: {p}")
        items.append(p)
    if not items:
        raise ValueError(f"Queue file is empty: {path}")
    return items


def ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        return sync_playwright
    except Exception as exc:  # pragma: no cover - only hit when dependency missing.
        raise RuntimeError(
            "Missing dependency: playwright. Install it in the lbb env with:\n"
            "  python -m pip install -r requirements.submit.txt\n"
            "  python -m playwright install chromium\n"
        ) from exc


def click_first_text(page, regex: re.Pattern[str], timeout_ms: int = 1500) -> bool:
    candidates = [
        page.get_by_role("button", name=regex),
        page.get_by_role("link", name=regex),
        page.locator("button").filter(has_text=regex),
        page.locator("a").filter(has_text=regex),
        page.locator("text=").filter(has_text=regex),
    ]
    for locator in candidates:
        try:
            if locator.count() > 0:
                locator.first.click(timeout=timeout_ms)
                return True
        except Exception:
            continue
    return False


def click_text_exact_or_regex(page, text: str, regex: Optional[re.Pattern[str]] = None, timeout_ms: int = 3000) -> bool:
    """Click a visible element by exact text first, then optional regex fallback."""
    locators = [
        page.get_by_text(text, exact=True),
        page.locator("li").filter(has_text=text),
        page.locator("span").filter(has_text=text),
        page.locator("div").filter(has_text=text),
    ]
    for locator in locators:
        try:
            if locator.count() > 0:
                locator.first.click(timeout=timeout_ms)
                return True
        except Exception:
            continue
    if regex is not None:
        return click_first_text(page, regex, timeout_ms=timeout_ms)
    return False


def first_usable_input(page, selectors: List[str]):
    for selector in selectors:
        loc = page.locator(selector)
        try:
            count = loc.count()
        except Exception:
            continue
        for i in range(count):
            item = loc.nth(i)
            try:
                if item.is_visible(timeout=500) and item.is_enabled(timeout=500):
                    return item
            except Exception:
                continue
    return None


def ensure_agreement_checked(page) -> None:
    """Best-effort click for Nowcoder's login agreement checkbox."""
    selectors = [
        ".sparta-login-form-footer .el-checkbox__input:not(.is-checked)",
        ".sparta-login-form-footer .el-checkbox",
        "label.el-checkbox",
    ]
    for selector in selectors:
        loc = page.locator(selector)
        try:
            if loc.count() > 0:
                loc.first.click(timeout=1000)
                page.wait_for_timeout(300)
                return
        except Exception:
            continue


def ensure_login_dialog_open(page) -> None:
    """Open the Nowcoder login dialog if it is not already open."""
    dialog = page.locator(".el-dialog__wrapper")
    try:
        if dialog.count() > 0 and dialog.first.is_visible(timeout=500):
            return
    except Exception:
        pass

    # Prefer the explicit header login button. `force=True` avoids a race where
    # the modal appears while Playwright is still waiting for pointer stability.
    for selector in [".loginBtnText", "text=登录/注册"]:
        loc = page.locator(selector)
        try:
            if loc.count() > 0:
                loc.first.click(timeout=3000, force=True)
                page.wait_for_timeout(1000)
                return
        except Exception:
            continue

    click_first_text(page, re.compile(r"(登录|密码登录|账号登录|Log ?in|Sign ?in)", re.I), timeout_ms=3000)
    page.wait_for_timeout(1000)


def auto_login(page, args) -> bool:
    account = os.environ.get(args.account_env, "").strip()
    password = os.environ.get(args.password_env, "").strip()
    if not account or not password:
        raise RuntimeError(
            f"--auto-login 需要环境变量 {args.account_env} 和 {args.password_env}。"
            "脚本只读取它们，不会写入文件。"
        )
    print(f"自动登录使用账号: {mask_account(account)}", flush=True)
    if account.isdigit() and len(account) != 11:
        eprint("提示: 当前账号是纯数字但不是 11 位手机号，牛客可能会提示“请输入正确的手机号码”。")

    # Open the login dialog if the page does not already show login inputs.
    ensure_login_dialog_open(page)

    # Nowcoder opens the SMS/register tab by default. Switch to password login
    # before looking for input[type=password].
    click_text_exact_or_regex(page, "密码登录", re.compile(r"密码登录|账号密码", re.I), timeout_ms=3000)
    page.wait_for_timeout(1200)

    account_input = first_usable_input(
        page,
        [
            "input[type='tel']",
            "input[name*='phone' i]",
            "input[name*='mobile' i]",
            "input[name*='account' i]",
            "input[placeholder*='手机']",
            "input[placeholder*='账号']",
            "input[placeholder*='邮箱']",
            "input[placeholder*='手机号']",
            "input[type='text']",
        ],
    )
    password_input = first_usable_input(
        page,
        [
            "input[type='password']",
            "input[placeholder*='密码']",
        ],
    )
    if account_input is None or password_input is None:
        save_debug_artifacts(page, args.work_dir, "auto_login_no_inputs")
        return False

    account_input.fill(account)
    password_input.fill(password)
    page.wait_for_timeout(500)
    ensure_agreement_checked(page)

    if not click_first_text(page, re.compile(r"(登录|立即登录|Sign ?in|Log ?in)", re.I), timeout_ms=5000):
        save_debug_artifacts(page, args.work_dir, "auto_login_no_button")
        return False

    page.wait_for_load_state("networkidle", timeout=args.nav_timeout_ms)
    page.wait_for_timeout(3000)
    save_debug_artifacts(page, args.work_dir, "auto_login_after_submit")

    # If the modal is still open, report visible form errors before deciding
    # whether a real verification challenge appeared.
    visible_text = ""
    try:
        visible_text = page.locator("body").inner_text(timeout=3000)
    except Exception:
        visible_text = ""

    err = LOGIN_ERROR_RE.search(visible_text)
    if err:
        eprint(f"登录表单提示: {err.group(1)}")
        return False

    # Successful login normally closes the login dialog or changes the header.
    try:
        dialog_visible = page.locator(".el-dialog__wrapper").first.is_visible(timeout=1000)
    except Exception:
        dialog_visible = False
    if not dialog_visible:
        return True

    body = ""
    try:
        body = page.locator("body").inner_text(timeout=3000)
    except Exception:
        pass
    if LOGIN_CHALLENGE_RE.search(body):
        eprint("页面疑似出现验证码/安全验证，脚本不会尝试绕过。请改用可见浏览器手动登录。")
        return False
    eprint("登录后弹窗仍未关闭，可能账号/密码未通过或页面需要人工确认。")
    return False


def find_file_input(page, explicit_selector: Optional[str] = None):
    if explicit_selector:
        loc = page.locator(explicit_selector)
        if loc.count() == 0:
            raise RuntimeError(f"Explicit file input selector matched nothing: {explicit_selector}")
        return loc.first

    loc = page.locator("input[type='file']")
    if loc.count() > 0:
        return loc.first

    # Some pages create the input only after an upload button is clicked.
    click_first_text(page, UPLOAD_BUTTON_RE)
    page.wait_for_timeout(500)
    loc = page.locator("input[type='file']")
    if loc.count() > 0:
        return loc.first

    return None


def click_submit(page, explicit_selector: Optional[str] = None) -> bool:
    if explicit_selector:
        page.locator(explicit_selector).first.click(timeout=5000)
        return True
    return click_first_text(page, SUBMIT_BUTTON_RE, timeout_ms=5000)


def click_tab(page, tab_name: str, timeout_ms: int = 10000) -> None:
    locators = [
        page.get_by_role("tab", name=tab_name),
        page.get_by_text(tab_name, exact=True),
        page.locator(".el-tabs__item").filter(has_text=tab_name),
    ]
    last_error: Optional[Exception] = None
    for loc in locators:
        try:
            if loc.count() > 0:
                loc.first.click(timeout=timeout_ms)
                page.wait_for_timeout(1200)
                return
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"找不到或无法点击 tab: {tab_name}. last_error={last_error}")


def ensure_competition_answer_tab(page) -> None:
    click_tab(page, "竞赛答题")
    try:
        page.locator("text=我的回答").wait_for(timeout=10000)
    except Exception as exc:
        save_debug_artifacts(page, DEFAULT_WORK_DIR.resolve(), "answer_tab_missing_my_answer")
        raise RuntimeError("进入竞赛答题页后没有看到“我的回答”，可能登录态失效或页面结构变化。") from exc


def select_nowcoder_task(page, task_name: str) -> None:
    """Select the target task in the '我的回答' section."""
    # If already selected, the select input value contains the task name.
    selected = page.locator(".el-select .el-input__inner").filter(has_text=task_name)
    try:
        if selected.count() > 0:
            return
    except Exception:
        pass

    label = page.get_by_text("请选择要回答的赛题", exact=True)
    if label.count() == 0:
        raise RuntimeError("找不到“请选择要回答的赛题”文本，无法选择赛题。")

    # The select box is the first following .el-select after the label.
    select_box = label.first.locator("xpath=following-sibling::div[contains(@class, 'el-select')][1]")
    if select_box.count() == 0:
        select_box = page.locator(".el-select").filter(has=page.locator("input[placeholder='请选择']")).first

    select_box.click(timeout=10000)
    page.wait_for_timeout(800)

    option = page.locator(".el-select-dropdown__item").filter(has_text=task_name)
    count = option.count()
    if count == 0:
        raise RuntimeError(f"下拉框里找不到赛题: {task_name}")

    # Element-UI keeps hidden dropdowns in DOM; click the last matching option,
    # which is usually the currently opened dropdown.
    clicked = False
    for i in reversed(range(count)):
        item = option.nth(i)
        try:
            item.click(timeout=3000)
            clicked = True
            break
        except Exception:
            continue
    if not clicked:
        raise RuntimeError(f"无法点击赛题选项: {task_name}")
    page.wait_for_timeout(1200)

    try:
        page.locator("input.el-input__inner").filter(has_text=task_name).first.wait_for(timeout=3000)
    except Exception:
        # Not all input values appear as text to Playwright; rely on upload
        # button state below for the stronger check.
        pass


def upload_nowcoder_answer_zip(page, zip_path: Path, args) -> None:
    """Nowcoder-specific answer submission flow.

    The real flow is:
    1. Enter '竞赛答题'
    2. Select the target task in '我的回答'
    3. Trigger the Element-UI upload input next to '上传附件'
    4. Open '提交记录' to verify a new row appears
    """
    ensure_competition_answer_tab(page)
    select_nowcoder_task(page, args.task_name)

    upload_area = page.locator(".upload-demo").first
    if upload_area.count() == 0:
        save_debug_artifacts(page, args.work_dir, f"no_upload_area_{zip_path.stem}")
        raise RuntimeError("没有找到“上传附件”区域。")

    upload_button = upload_area.locator("button").first
    try:
        cls = upload_button.get_attribute("class") or ""
        disabled = upload_button.get_attribute("disabled")
    except Exception:
        cls, disabled = "", None
    if disabled is not None or "is-disabled" in cls:
        save_debug_artifacts(page, args.work_dir, f"upload_button_disabled_{zip_path.stem}")
        raise RuntimeError("已进入答题页，但“上传附件”按钮仍是 disabled，赛题可能没有选中。")

    file_input = upload_area.locator("input[type='file'][accept='.zip']").first
    if file_input.count() == 0:
        file_input = upload_area.locator("input[type='file']").first
    if file_input.count() == 0:
        save_debug_artifacts(page, args.work_dir, f"no_answer_file_input_{zip_path.stem}")
        raise RuntimeError("上传附件区域里没有找到 file input。")

    before_text = ""
    try:
        before_text = page.locator("body").inner_text(timeout=3000)
    except Exception:
        pass

    file_input.set_input_files(str(zip_path))
    print("文件已选择并触发上传附件。", flush=True)
    page.wait_for_timeout(int(args.after_upload_wait_ms))

    # Some Element-UI upload flows show a confirmation dialog or toast. Click a
    # conservative positive button if one appears.
    for pattern in [re.compile(r"^(确定|确认)$"), re.compile(r"(提交|上传)$")]:
        try:
            btn = page.get_by_role("button", name=pattern)
            if btn.count() > 0 and btn.first.is_visible(timeout=1000):
                btn.first.click(timeout=3000)
                page.wait_for_timeout(1000)
                break
        except Exception:
            continue

    page.wait_for_timeout(int(args.after_submit_wait_ms))
    click_tab(page, "提交记录")
    page.wait_for_timeout(2500)

    body = page.locator("body").inner_text(timeout=5000)
    if zip_path.name not in body:
        save_debug_artifacts(page, args.work_dir, f"upload_not_in_records_{zip_path.stem}")
        # Keep this as an error: otherwise it is too easy to think a ZIP was
        # submitted while the page merely displayed old public records.
        raise RuntimeError(
            f"提交记录里没有看到刚上传的文件名: {zip_path.name}。"
            "可能上传失败、登录态失效，或页面仍停留在错误赛题。"
        )
    if before_text and zip_path.name in before_text:
        print("提交记录中原本已存在同名文件；已重新检查到该文件名。", flush=True)
    else:
        print("提交记录已出现刚上传的文件名。", flush=True)


def save_debug_artifacts(page, work_dir: Path, prefix: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)[:120]
    try:
        page.screenshot(path=str(work_dir / f"{safe}.png"), full_page=True)
    except Exception:
        pass
    try:
        (work_dir / f"{safe}.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def latest_scores_from_page(page) -> List[str]:
    try:
        text = page.locator("body").inner_text(timeout=3000)
    except Exception:
        return []
    return SCORE_RE.findall(text)


def launch_browser(playwright, args):
    launch_kwargs = {
        "headless": bool(args.headless),
        "slow_mo": int(args.slow_mo),
    }
    if args.browser_executable:
        launch_kwargs["executable_path"] = args.browser_executable

    storage_state = str(args.state) if args.state.exists() else None
    browser = playwright.chromium.launch(**launch_kwargs)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    return browser, context, page


def page_looks_logged_in(page) -> bool:
    """Heuristic check for whether the current Nowcoder page is logged in."""
    try:
        dialog = page.locator(".el-dialog__wrapper")
        if dialog.count() > 0 and dialog.first.is_visible(timeout=500):
            return False
    except Exception:
        pass

    login_selectors = [
        ".loginBtnText",
        "text=登录/注册",
    ]
    for selector in login_selectors:
        loc = page.locator(selector)
        try:
            if loc.count() > 0 and loc.first.is_visible(timeout=500):
                return False
        except Exception:
            continue

    try:
        text = page.locator("body").inner_text(timeout=3000)
    except Exception:
        text = ""
    if "登录/注册" in text and "密码登录" in text:
        return False
    return True


def do_check_login(args) -> None:
    if not args.state.exists():
        raise FileNotFoundError(f"登录态文件不存在: {args.state}")

    sync_playwright = ensure_playwright()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, context, page = launch_browser(p, args)
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=args.nav_timeout_ms)
            page.wait_for_load_state("networkidle", timeout=args.nav_timeout_ms)
            save_debug_artifacts(page, args.work_dir, "check_login")
            if page_looks_logged_in(page):
                print(f"登录态看起来有效: {args.state}")
            else:
                raise RuntimeError(
                    "登录态看起来无效：页面仍显示登录入口或登录弹窗。"
                    "请重新在本地图形界面导出 state.json。"
                )
        finally:
            browser.close()


def do_login(args) -> None:
    sync_playwright = ensure_playwright()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, context, page = launch_browser(p, args)
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.nav_timeout_ms)
        if args.auto_login:
            ok = auto_login(page, args)
            if not ok:
                browser.close()
                raise RuntimeError(
                    "自动登录没有成功。若平台要求验证码/扫码，请在有图形界面的机器上运行 --login 手动登录。"
                )
        else:
            if args.headless:
                browser.close()
                raise RuntimeError("--login 需要可见浏览器；headless 模式请搭配 --auto-login。")
            print(
                "\n已打开牛客页面。请在浏览器里手动登录/进入提交页面，完成后回到终端按 Enter。\n"
                "如果页面有验证码，请手动完成，不要让脚本绕过验证码。\n",
                flush=True,
            )
            input("登录完成后按 Enter 保存登录态...")
        context.storage_state(path=str(args.state))
        save_debug_artifacts(page, args.work_dir, "login_saved")
        browser.close()
    print(f"登录态已保存: {args.state}")


def submit_one(page, zip_path: Path, args, idx: int, total: int) -> None:
    print(f"\n[{idx}/{total}] 准备提交: {zip_path.name}", flush=True)
    page.goto(args.url, wait_until="domcontentloaded", timeout=args.nav_timeout_ms)
    page.wait_for_load_state("networkidle", timeout=args.nav_timeout_ms)

    if args.nowcoder_exam_flow:
        if args.confirm_each:
            input("请确认要按牛客答题页流程上传该 ZIP，按 Enter 继续...")
        upload_nowcoder_answer_zip(page, zip_path, args)
        save_debug_artifacts(page, args.work_dir, f"submitted_{idx:03d}_{zip_path.stem}")
        return

    file_input = find_file_input(page, args.file_input_selector)
    if file_input is None:
        save_debug_artifacts(page, args.work_dir, f"no_file_input_{zip_path.stem}")
        raise RuntimeError(
            "没有找到文件上传控件。请先用 --login 打开页面确认已进入提交页；"
            "如果页面按钮文案特殊，可以加 --file-input-selector 指定 CSS 选择器。"
        )

    file_input.set_input_files(str(zip_path))
    page.wait_for_timeout(int(args.after_upload_wait_ms))
    print("文件已选择。", flush=True)

    if args.confirm_each:
        input("请检查浏览器页面，确认要提交后按 Enter...")

    if not args.no_click_submit:
        clicked = click_submit(page, args.submit_selector)
        if not clicked:
            save_debug_artifacts(page, args.work_dir, f"no_submit_button_{zip_path.stem}")
            raise RuntimeError(
                "文件已选择，但没有找到提交按钮。可以手动点提交，或用 --submit-selector 指定 CSS 选择器。"
            )
        print("已点击提交按钮。", flush=True)
    else:
        print("已选择文件，但按 --no-click-submit 要求没有自动点击提交。", flush=True)

    page.wait_for_timeout(int(args.after_submit_wait_ms))
    if args.wait_score_seconds > 0:
        deadline = time.time() + float(args.wait_score_seconds)
        seen: List[str] = []
        while time.time() < deadline:
            scores = latest_scores_from_page(page)
            if scores and scores != seen:
                seen = scores
                print(f"页面检测到分数字符串: {', '.join(scores[-5:])}", flush=True)
            page.wait_for_timeout(5000)

    save_debug_artifacts(page, args.work_dir, f"submitted_{idx:03d}_{zip_path.stem}")


def do_submit(args) -> None:
    queue = read_queue(args.queue)
    print("提交队列：")
    for i, p in enumerate(queue, 1):
        print(f"  {i}. {p}")

    if args.dry_run:
        print("\n--dry-run：只检查队列，不打开浏览器。")
        return

    sync_playwright = ensure_playwright()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, context, page = launch_browser(p, args)
        try:
            for i, zip_path in enumerate(queue, 1):
                submit_one(page, zip_path, args, i, len(queue))
                if i < len(queue):
                    print(f"等待 {args.interval_seconds} 秒后提交下一个...", flush=True)
                    page.wait_for_timeout(int(args.interval_seconds * 1000))
            context.storage_state(path=str(args.state))
        finally:
            if args.keep_open:
                input("提交流程结束。浏览器保持打开，按 Enter 关闭...")
            browser.close()


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Submit LBB ZIP queue to Nowcoder.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--login", action="store_true", help="Open browser for manual login and save state.")
    parser.add_argument("--check-login", action="store_true", help="Check whether the saved storage state is logged in.")
    parser.add_argument("--auto-login", action="store_true", help="Login from env vars, useful on headless servers.")
    parser.add_argument("--account-env", default="NOWCODER_ACCOUNT")
    parser.add_argument("--password-env", default="NOWCODER_PASSWORD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Run browser headless after login state exists.")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--confirm-each", action="store_true", help="Ask before clicking submit for every ZIP.")
    parser.add_argument("--no-click-submit", action="store_true", help="Only select ZIP file, do not click submit.")
    parser.add_argument("--file-input-selector", default=None)
    parser.add_argument("--submit-selector", default=None)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--generic-upload", action="store_true", help="Use old generic file-input upload flow.")
    parser.add_argument("--browser-executable", default=None)
    parser.add_argument("--slow-mo", type=int, default=80)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--after-upload-wait-ms", type=int, default=1500)
    parser.add_argument("--after-submit-wait-ms", type=int, default=5000)
    parser.add_argument("--wait-score-seconds", type=float, default=0.0)
    parser.add_argument("--nav-timeout-ms", type=int, default=60000)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.state = args.state.resolve()
    args.queue = args.queue.resolve()
    args.nowcoder_exam_flow = not bool(args.generic_upload)

    try:
        if args.check_login:
            do_check_login(args)
        elif args.login:
            do_login(args)
        else:
            do_submit(args)
    except KeyboardInterrupt:
        eprint("\n用户中断。")
        return 130
    except Exception as exc:
        eprint(f"\n提交 agent 出错: {exc}")
        eprint(f"调试截图/HTML 如有生成，会保存在: {args.work_dir}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
