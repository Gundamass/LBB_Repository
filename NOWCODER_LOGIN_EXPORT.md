# 牛客登录态本地导出流程

这个流程用于处理服务器无图形界面、牛客登录需要验证码/滑块/扫码的情况。

核心思路：

1. 在你自己的有图形界面的电脑上打开浏览器，手动完成牛客登录和验证码。
2. 导出 Playwright 的 `state.json`。
3. 把 `state.json` 放回服务器。
4. 服务器复用这个登录态，headless 自动提交 ZIP。

## 1. 在本地电脑准备导出工具

从服务器取回本地登录导出包：

```bash
scp heqing@amax:/home/heqing/LBB_competition/outputs/nowcoder_login_export_bundle.zip .
```

解压：

```bash
unzip nowcoder_login_export_bundle.zip -d nowcoder_login_export_bundle
cd nowcoder_login_export_bundle
```

安装依赖：

```bash
python -m pip install -r requirements.submit.txt
python -m playwright install chromium
```

## 2. 本地打开浏览器并导出 state.json

```bash
python export_nowcoder_state.py --state state.json
```

浏览器打开后，请手动完成：

1. 点击 `登录/注册`
2. 切换到 `密码登录`
3. 输入账号和密码
4. 勾选同意协议
5. 完成验证码/滑块/扫码验证
6. 确认页面已经登录成功：右上角不再显示 `登录/注册`，而是头像/用户状态
7. 最好停留在比赛 `竞赛答题` 页面
8. 回到终端按 Enter

如果脚本提示“页面看起来仍未登录”，请不要强制导出；继续在浏览器里完成登录和验证码，确认右上角已经变成头像后再按 Enter。

成功后当前目录会生成：

```text
state.json
```

## 3. 把 state.json 拷回服务器

```bash
scp state.json heqing@amax:/home/heqing/LBB_competition/outputs/.nowcoder_submitter/state.json
```

如果服务器地址不是 `amax`，把命令里的主机名换成你实际 SSH 地址。

## 4. 在服务器检查登录态

```bash
cd /home/heqing/LBB_competition
conda activate lbb

python tools/nowcoder_submit_agent.py --headless --check-login
```

如果输出类似下面这样，就可以提交：

```text
登录态看起来有效: /home/heqing/LBB_competition/outputs/.nowcoder_submitter/state.json
```

如果提示登录态无效，需要重新在本地导出一次。

如果导入成功，提交 agent 会走牛客答题页专用流程：

1. 点击 `竞赛答题`
2. 在 `我的回答` 中选择 `少样本条件下电子产品外观缺陷检测`
3. 点击/触发 `上传附件`
4. 上传队列中的 ZIP
5. 切换到 `提交记录` 检查是否出现刚上传的文件名

## 5. 服务器自动提交队列

先检查队列：

```bash
python tools/nowcoder_submit_agent.py --dry-run
```

半自动提交，每个 ZIP 前暂停确认：

```bash
python tools/nowcoder_submit_agent.py --headless --confirm-each --interval-seconds 45
```

确认流程稳定后，可以全自动提交当前队列：

```bash
python tools/nowcoder_submit_agent.py --headless --interval-seconds 45
```

当前队列文件：

```text
submit_queue_nowcoder.txt
```

建议每轮只放 2-3 个 ZIP，提交完看线上分数后再决定下一轮。
