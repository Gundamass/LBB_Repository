#!/usr/bin/env python3
"""Export Nowcoder Playwright login state on a local GUI machine.

Run this on a machine where you can see and operate the browser. It opens
Nowcoder, lets you manually finish password login / captcha / slider, then
saves a Playwright storage_state JSON file that can be copied back to the
server for headless submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_URL = "https://competition.nowcoder.com/exam/226/419"


def page_looks_logged_in(page) -> bool:
    try:
        dialog = page.locator(".el-dialog__wrapper")
        if dialog.count() > 0 and dialog.first.is_visible(timeout=500):
            return False
    except Exception:
        pass
    for selector in [".loginBtnText", "text=登录/注册"]:
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
    return "登录/注册" not in text or "竞赛答题" in text


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Nowcoder login state.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--state", type=Path, default=Path("state.json"))
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default="chromium")
    parser.add_argument("--slow-mo", type=int, default=80)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: playwright. Install with:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium\n"
        ) from exc

    state_path = args.state.expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser_type = getattr(p, args.browser)
        browser = browser_type.launch(headless=False, slow_mo=int(args.slow_mo))
        context = browser.new_context()
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)

        print(
            "\n浏览器已打开。请在页面里手动完成：\n"
            "1. 点击 登录/注册\n"
            "2. 切换到 密码登录\n"
            "3. 输入账号密码并勾选同意协议\n"
            "4. 完成验证码/滑块/扫码等验证\n"
            "5. 确认已经登录成功，最好停留在比赛提交页面\n",
            flush=True,
        )
        while True:
            input("登录成功后回到这里按 Enter，脚本会检查并导出 state.json...")
            if page_looks_logged_in(page):
                break
            answer = input(
                "页面看起来仍未登录，可能还在登录弹窗或右上角仍显示“登录/注册”。"
                "请继续在浏览器完成登录后按 Enter；如果仍要强制导出，输入 yes: "
            ).strip().lower()
            if answer == "yes":
                break
        context.storage_state(path=str(state_path))
        browser.close()

    print(f"\n已导出登录态: {state_path}")
    print("请把这个 state.json 复制到服务器路径：")
    print("/home/heqing/LBB_competition/outputs/.nowcoder_submitter/state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
