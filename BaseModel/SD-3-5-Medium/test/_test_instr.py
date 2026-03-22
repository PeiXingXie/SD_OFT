"""
Quick test instructions (run from repo root after starting the daemon server):

1) Start server (in another terminal):
  python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_server.py --host 127.0.0.1 --port 6319 --device cuda --torch_dtype bfloat16

2) Run a few client calls:
  python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py --host 127.0.0.1 --port 6319 --prompt "A capybara holding a sign that reads Hello World, in Pointillism Style." --out BaseModel/SD-3-5-Medium/test/capybara-Pointillism.png
  python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py --host 127.0.0.1 --port 6319 --prompt "a painting of a church with a clock tower, in Baroque Style." --out BaseModel/SD-3-5-Medium/test/church-Baroque.png
"""