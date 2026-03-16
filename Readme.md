# H-optimus-0-valis-lora

module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/H-optimus-0-valis-lora/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/H-optimus-0-valis-lora/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list