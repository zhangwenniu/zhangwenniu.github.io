---
layout: book
title: 第二章：Python基础
book_id: book1
book_title: 编程入门指南
book_description: 不太一样的描述。
chapter_number: 2
previous_chapter: /books/book1/ch1
date: 2023-01-02
---

## 2.1 Python简介

Python是一种高级编程语言，以其简洁、易读的语法而闻名。它由Guido van Rossum于1991年创建，现已成为全球最流行的编程语言之一。Python强调代码的可读性，其语法允许程序员用更少的代码行表达概念。

## 2.2 安装Python

在开始编写Python代码之前，你需要在你的计算机上安装Python。

### Windows系统
1. 访问Python官方网站 [python.org](https://www.python.org/downloads/)
2. 下载最新版本的Python安装程序
3. 运行安装程序，确保勾选"Add Python to PATH"选项
4. 完成安装

### macOS系统
1. macOS通常预装了Python，但可能不是最新版本
2. 你可以通过Homebrew安装最新版本：`brew install python`

### Linux系统
1. 大多数Linux发行版预装了Python
2. 你可以通过包管理器安装最新版本，例如Ubuntu：`sudo apt-get install python3`

## 2.3 第一个Python程序

让我们编写一个简单的"Hello, World!"程序，这是学习任何编程语言的传统第一步。

1. 打开你喜欢的文本编辑器或IDE
2. 创建一个新文件，命名为`hello.py`
3. 输入以下代码：

```python
print("Hello, World!")
```

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### 8.1 基本接口

我们首先将为单个的双向反射分布函数（BRDF）和双向透射分布函数（BTDF）定义接口。BRDF 和 BTDF 共享一个公共基类 BxDF。由于它们具有完全相同的接口，共享相同的基类可以减少重复代码，并使系统的某些部分能够通用地处理 BxDF，而无需区分 BRDF 和 BTDF。


```cpp
<BxDF 声明>≡
class BxDF {
public:
〈BxDF 接口 513〉
〈BxDF 公共数据 513〉
};
```


4. 保存文件
5. 打开命令行或终端
6. 导航到保存文件的目录
7. 运行命令：`python hello.py`

如果一切正常，你应该会在屏幕上看到"Hello, World!"输出。恭喜！你已经成功编写并运行了你的第一个Python程序。

## 2.4 Python变量和数据类型

在Python中，你可以使用变量来存储数据。Python是动态类型的，这意味着你不需要声明变量的类型。

### 变量赋值
```python
name = "张三"
age = 25
height = 1.75
is_student = True
```

### 基本数据类型
- **字符串(str)**：文本数据，如`"Hello"`
- **整数(int)**：如`42`
- **浮点数(float)**：如`3.14`
- **布尔值(bool)**：`True`或`False`
- **列表(list)**：有序集合，如`[1, 2, 3]`
- **元组(tuple)**：不可变的有序集合，如`(1, 2, 3)`
- **字典(dict)**：键值对集合，如`{"name": "张三", "age": 25}`

## 2.5 基本操作

### 算术运算
```python
a = 10
b = 3

print(a + b)  # 加法: 13
print(a - b)  # 减法: 7
print(a * b)  # 乘法: 30
print(a / b)  # 除法: 3.3333...
print(a // b) # 整除: 3
print(a % b)  # 取余: 1
print(a ** b) # 幂运算: 1000
```

### 字符串操作
```python
first_name = "张"
last_name = "三"

full_name = first_name + last_name  # 字符串连接
print(full_name)  # 输出: 张三

greeting = "你好，" + full_name
print(greeting)  # 输出: 你好，张三

repeated = "哈" * 3
print(repeated)  # 输出: 哈哈哈
```

在下一章中，我们将学习条件语句和循环，这些是编程中的基本控制结构。 