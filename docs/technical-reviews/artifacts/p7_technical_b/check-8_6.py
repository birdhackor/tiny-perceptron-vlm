import json
requests=[{"task":"確認日期","date":None},{"task":"確認日期","date":"2026-10-03"}]
out=[]
for request in requests:
 target="你希望哪一天？" if request["date"] is None else "已確認日期："+request["date"]
 print(request,"→",target);out.append({"request":request,"target":target})
print(json.dumps({"original":out,"variation_both_given":[{"date":d,"target":"已確認日期："+d} for d in ["2026-10-04","2026-10-03"]],"blank_is_none":"" is None},ensure_ascii=False))
