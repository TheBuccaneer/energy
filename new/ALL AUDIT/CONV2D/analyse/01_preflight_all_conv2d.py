#!/usr/bin/env python3
from __future__ import annotations
import json
from all_conv2d_common import *

def main() -> None:
    project,result,_=roots(__file__)
    rows=[]; failed=False
    for p in PLATFORMS:
        r=project/p/"results"/"CONV2D"
        required=["ANALYSIS_COMPLETE.json","configuration_summary.csv","session_medians.csv","validation_checks.csv"]
        for name in required:
            present=(r/name).is_file(); rows.append({"platform":p,"artifact":name,"status":"PASS" if present else "FAIL"})
            failed |= not present
        if (r/"validation_checks.csv").is_file():
            checks=pd.read_csv(r/"validation_checks.csv")
            hard=((checks.severity=="FAIL")&(checks.status=="FAIL")).sum()
            rows.append({"platform":p,"artifact":"hard_validation_failures","status":"PASS" if hard==0 else "FAIL","observed":int(hard)})
            failed |= hard>0
    pd.DataFrame(rows).to_csv(result/"preflight_checks.csv",index=False)
    (result/"preflight_complete.json").write_text(json.dumps({"status":"FAIL" if failed else "PASS"},indent=2),encoding="utf-8")
    print("[CONV2D all-platform] preflight:","FAIL" if failed else "PASS")
    if failed: raise SystemExit(2)
if __name__=="__main__": main()
