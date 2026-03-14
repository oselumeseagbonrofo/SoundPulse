# Audio data dataset classes

Audio data dataset location

ESC-50 dataset location - ml\\data\\raw\\esc\_50\_dataset

FSD50K dataset location - ml\\data\\raw\\FSDK50\_dataset

| Class Name | Name in datasets | Dataset contained in |
| --- | --- | --- |
| alarm | alarm | FSD50k (/m/07pp\_mv) |
| alarm clock | clock\_alarm (ESC-50) | ESC-50 |
| door knock | Knock (FSD50k)/ door\_wood\_knock (ESC-50) | FSD50k (/m/07r4wb8) ESC-50 |
| doorbell | doorbell | FSD50k (/m/03wwcy) |
| Crying baby | Crying\_and\_sobbing (FSD50K)/crying\_baby (ESC-50) | FSD50K (/m/0463cq4), ESC-50 |
| water running | Water\_tap\_and\_faucet (FSD50k) | FSD50K (/m/02jz0l) |
| dog barking | bark (FSD50K)/ dog (ESC-50) | FSD50K(/m/05tny\_), ESC-50 |
| door wood creaking | door\_wood\_creaks (ESC-50) / Squeak (FSD50K) | FSD50K (/m/07q6cd\_), ESC-50 |
| speech | speech (FSD50K) | FSD50K (/m/09x0r) |
| null background class | Combination of dishes\_and\_pots\_and\_pans (/m/04brg2), Frying\_(food) (/m/0dxrf), |   |
| Music (/m/04rlf), mechanical\_fan (/m/02x984l) |   |   |
| mixed randomly with other sound class | FSD50k |   |

**Note:** for FSD50K data, the value in bracket in the "Dataset contained in" column corresponds to the class\_id in ml\\data\\raw\\FSDK50\_dataset\\FSD50K.ground\_truth\\vocabulary.csv