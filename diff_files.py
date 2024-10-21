# -*- coding: utf-8 -*-


import os
target = "plugging"
if target == "tensorrt":
    template_folder = "/usr/local/lib/python3.10/dist-packages/tensorrt_llm"
    new_folder = "/code/tensorrt_llm/tensorrt_llm"
if target == "modelopt":
    template_folder = "/usr/local/lib/python3.10/dist-packages/modelopt"
    new_folder = "/code/tensorrt_llm/modelopt"
if target == "plugging":
    template_folder = "/usr/local/lib/python3.10/dist-packages/kernel"
    new_folder = "/code/tensorrt_llm/kernel"
import os

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


num = 0
out = ""
os.system("rm -rf out.txt")
cnt = 13
#cnt = 7
for  i in findAllFile(new_folder):
    if  i.endswith("py") or i.endswith(".cpp") or i.endswith(".cu") or i.endswith(".h") or i.endswith(".hpp"):
        if ("cutlass" in i):
            continue
        print(i)
        source = i.replace(new_folder,template_folder)
        if "plugging" in target  :
            source = "empty.py"
        tmp = os.popen("diff " + i  + " " +  source )
        tmp = tmp.read()
        print(len(tmp))
        if len(tmp):
            cnt += 1
            out += ("## " + str(cnt) + ". 文件" + i.replace(new_folder,"") + " 的修改内容为："+ "\n")

            out += ("``` " + tmp + "``` " + "\n")
        
f = open("out.txt","w+")
f.writelines(out)
f.close()

 