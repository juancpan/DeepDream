# DeepDream
```
Pytorch Implementation of Google-DeepDream
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.
```

# Introduction
#### in Chinese
https://mp.weixin.qq.com/s/iIhiMKutVtYEUgAiErLkVQ

# Environment
```
OS: Ubuntu 16.04
Graphics card: Titan xp
Python: python3.x with the packages in requirements.txt
```

# Usage
```
# uncontrolled
python train.py --imagepath imgs/sky1024px.jpg
# controlled
python train.py --imagepath imgs/sky1024px.jpg --controlimagepath imgs/flowers.jpg --iscontrolled
```

# Results
#### uncontrolled
![img](./docs/unsupervise.jpg)
#### controlled
![img](./docs/supervise.jpg)

# Reference
```
[1]. https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
[2]. https://github.com/google/deepdream
```

# More
#### WeChat Official Accounts
*Charles_pikachu*  
![img](./docs/pikachu.jpg)