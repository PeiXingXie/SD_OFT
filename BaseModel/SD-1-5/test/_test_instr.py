"""
Quick test instructions (run from repo root after starting the daemon server):

1) Start server (in another terminal):
  python BaseModel/SD-1-5/sd1_5_daemon_server.py --host 127.0.0.1 --port 6315 --device cuda --torch_dtype float16

2) Run a few client calls:
  python BaseModel/SD-1-5/sd1_5_daemon_client.py --host 127.0.0.1 --port 6315 --prompt "A capybara holding a sign that reads Hello World, in Pointillism Style." --out BaseModel/SD-1-5/test/capybara-Pointillism.png
  python BaseModel/SD-1-5/sd1_5_daemon_client.py --host 127.0.0.1 --port 6315 --prompt "a painting of a church with a clock tower, in Baroque Style." --out BaseModel/SD-1-5/test/church-Baroque.png
"""