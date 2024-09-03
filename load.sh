
# pip install rouge_score
# pip install nltk
# pip install zstandard
# rm -r /usr/local/lib/python3.10/dist-packages/tensorrt_llm
cp .tmux.conf ~/.
pip install triton==2.3.0
#apt-get update
export PYTHONPATH=$PYTHONPATH:/code/tensorrt_llm/
export FT_LOG_LEVEL=ERROR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/code/tensorrt_llm/tensorrt_llm/libs
#cp -r /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs /code/tensorrt_llm/tensorrt_llm/libs
# pip install modelutils -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple
cd quantkernel
python setup.py install
cd ..

cd AutoAWQ 
python setup.py install
cd ..


cd EETQ
#rm -r build
python setup.py build
export PYTHONPATH=$PYTHONPATH:/code/tensorrt_llm/EETQ/build/lib.linux-x86_64-cpython-310

cd ..

#mkdir build
cd build
cmake ..
make
cd ..



pip install datasets==2.14.7
pip install -i https://pypi.org/simple/ bitsandbytes


# cd lm-evaluation-harness
# python setup.py install
# cd ..

pip install peft