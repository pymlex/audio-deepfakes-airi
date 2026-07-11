import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
_, o, _ = c.exec_command("grep -n 'epoch\\|Error\\|Traceback' /root/audio-deepfakes-airi/full_run.log | tail -40")
open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\full_log_grep.txt", "w", encoding="utf-8").write(o.read().decode("utf-8", errors="replace"))
c.close()
