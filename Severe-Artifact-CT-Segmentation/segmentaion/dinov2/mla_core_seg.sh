#!/bin/bash

echo "----------Training Starge--------------"
python ../train.py -net "mla" \
                   -d "core_seg" \
                   -dn "cuda" \
                   -v "small"\
                   -loss "wdice"\
                   -cp "lora"\
                   -l 1e-5
echo "----------Training Over----------------"

echo "---------------Evaluate----------------"
python ../test.py -net "mla" \
                    -d "core_seg" \
                    -dn "cuda" \
                    -v "small"\
                    -loss "wdice"\
                    -cp "lora"   #unfrozen | lora
echo "Done"
