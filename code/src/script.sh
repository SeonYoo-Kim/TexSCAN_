python main.py --m 50 --dataset 'DTD-sys'
python main.py --m 200 --dataset 'MVTec'
python main.py --m 250 --dataset 'BTAD'
python main.py --m 150 --dataset 'WFT'
python main.py --m 200 --dataset 'WFDD'

#for e in $(seq 20 5 40)
#do
#  for m in $(seq 1550 -50 50)
#  do
#    python main.py --e $e --m $m --dataset 'BTAD'
#  done
#done
#
#for e in $(seq 20 5 40)
#do
#  for m in $(seq 1550 -50 50)
#  do
#    python main.py --e $e --m $m --dataset 'WFDD'
#  done
#done
#
#for e in $(seq 20 5 40)
#do
#  for m in $(seq 1550 -50 50)
#  do
#    python main.py --e $e --m $m --dataset 'WFT'
#  done
#done