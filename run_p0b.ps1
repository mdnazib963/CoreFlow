$code = @'
import sys
sys.path.insert(0, "/content")
sys.argv = ["train.py", "--train", "/content/train.bin", "--val", "/content/val.bin", "--out", "/content/run_p0b", "--d", "144", "--dff", "384", "--nconv", "4", "--natt", "2", "--heads", "8", "--kv", "1", "--ctx", "1024", "--epochs", "3", "--max-steps", "17610", "--eval-every", "1000", "--threads", "2"]
exec(open("/content/train.py", encoding="utf-8").read())
'@
$code | colab exec -s gpu-t4 --timeout 3000 2>&1 | Out-File -FilePath D:\coreflow\run_p0b.log -Encoding utf8
"EXIT=$LASTEXITCODE" | Out-File -FilePath D:\coreflow\run_p0b.done -Encoding utf8
