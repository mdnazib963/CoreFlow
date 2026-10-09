$code = @'
import sys
sys.path.insert(0, "/content")
sys.argv = ["train.py", "--train", "/content/train.bin", "--val", "/content/val.bin", "--out", "/content/run_p0", "--d", "128", "--dff", "192", "--nconv", "4", "--natt", "2", "--heads", "4", "--kv", "1", "--ctx", "1024", "--max-steps", "12000", "--eval-every", "1000", "--threads", "2"]
exec(open("/content/train.py", encoding="utf-8").read())
'@
$code | colab exec -s gpu-t4 --timeout 3000 2>&1 | Out-File -FilePath D:\coreflow\run_p0.log -Encoding utf8
"EXIT=$LASTEXITCODE" | Out-File -FilePath D:\coreflow\run_p0.done -Encoding utf8
