import json

import pandas as pd


def create_sample():
    queries = ["敏感肌可以用？", "赠品有没有限名额"]
    sample_docs = [["敏感肌是可以使用的呢", "亲爱的，敏感肌也是可以放心使用的哦", "建议可以使用的呢"],
                   ["亲爱的K星人，骨头先生来发福利啦~10月21号-11月10日活动期间",
                    "预订牛油果眼霜 28g，即可获得4件K星礼：",
                    "1.金盏花爽肤水 40ml*12.金盏花精华水活霜 7ml*1",
                    "3.金盏花清透洁面啫喱 30ml*14.限量【限定K星环保袋】*1"]]
    sample_docs = [json.dumps([i]*5) for i in sample_docs]  # using json to store list of docs
    d = {"queries": queries, "docs": sample_docs}
    df = pd.DataFrame(data=d)
    df.to_csv("sample/sample.csv")


if __name__ == "__main__":
    create_sample()