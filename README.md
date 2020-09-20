# QQ桌球瞄准器
**基于图像处理的QQ桌球辅助器**

* 纯Python实现，基于图像处理算法
* 桌面提取，斯诺克，中式八球通用
* 完整的遮挡分析，支持组合球
* 袋口半遮挡避让，可行域计算，进球概率估算


![img](https://user-images.githubusercontent.com/24822467/93710301-23978000-fb78-11ea-9908-eac1c8f8ae19.png)

## 使用方法
**开启即可，当鼠标位于理论击球点附近，就可以自动吸附到正确位置。**
* 开始， 暂停，用于开启，关闭辅助瞄准功能

* 绘制可以提取当前图像，并展示击球策略

* 传击可以设定允许传递次数，1代表只允许1次，既禁止组合球

* 模式可以在斯诺克和中式八球互相切换

## 代码简介

### 图像提取模块：extract
提取画面，在hsv空间，利用区域分析，像素统计，形态学等方法，识别球桌，以及每个球的位置，类型。
![Figure_1](https://user-images.githubusercontent.com/24822467/93711266-732d7a00-fb7f-11ea-9fa9-e2dd856cd81f.png)

### 球桌对象及控件分析：table
table定义了点，线，面，球等基础几何对象，并在此基础上组装成Table对象，可以用于球的可视化与策略生成。
![s1](https://user-images.githubusercontent.com/24822467/93711457-1af77780-fb81-11ea-9531-287d7243c705.png)
生成红球击打策略，禁止组合球

![s2](https://user-images.githubusercontent.com/24822467/93711460-1e8afe80-fb81-11ea-8b8d-28d3542d5757.png)
生成红球击打策略，允许组合球

### 鼠标控制模块：robot
这个模块调用win32api，可以实现鼠标吸附。

### 应用程序界面：frame
使用tkinter搭建应用界面

![frame](https://user-images.githubusercontent.com/24822467/93711523-b7217e80-fb81-11ea-832e-8915f20bf327.png)

## 尚待完善
1. 算法比较通用，但是一些搜索参数可能会受到分辨率影响，目前只在自己的机器上做过测试。
2. 中式八球目前识别色球，花球，白球，黑球，是用简单的直方图，方差等信息，不是非常稳定。
3. 策略支持求解指定颜色，种类的球，但是目前尚未从画面中提取应该击打的目标球信息，因而没有区分球，做的通用求解。
4. 如果以上问题解决后，可以加入一个无需操作，全自动击球的功能。

**以上问题都不是非常复杂，但是需要在不同机器上测试，作为娱乐项目，不打算继续深入了，如果大家有兴趣，可以一起探讨，继续完善。**