$code = @'
import sys
sys.path.insert(0, "/content")
sys.argv = ["train.py", "--train", "/content/train.bin", "--val", "/content/val.bin", "--out", "/content/run3", "--eval-every", "500", "--threads", "2"]
exec(open("/content/train.py", encoding="utf-8").read())
'@
$code | colab exec -s gpu-t4 --timeout 3000 2>&1 | Out-File -FilePath D:\coreflow\run_full.log -Encoding utf8
"EXIT=$LASTEXITCODE" | Out-File -FilePath D:\coreflow\run_full.done -Encoding utf8
