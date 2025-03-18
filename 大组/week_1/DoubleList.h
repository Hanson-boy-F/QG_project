#define _CRT_SECURE_NO_WARNINGS

#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>

#ifndef DoubleLink

typedef struct DoubleLinkNode
{
	int data;
	struct DoubleLinkNode* front;
	struct DoubleLinkNode* next;
}LinkNode;
// 初始化链表
struct DoubleLinkNode* Linklist();
// 打印链表
void printLink(LinkNode* header);
// 反向打印链表
void reverseprintLink(LinkNode* header);
// 在指定位置插入节点
void insertLink(LinkNode* header, int oldvalue, int newvalue);
// 删除指定数据
void deleteLink(LinkNode** header, int deletevalue);
// 查找某个数据
void findLink(LinkNode* header, int value);
// 判断链表是否为空
void Empty(LinkNode* header);
// 获取链表长度
void Length(LinkNode* header);
// 清空链表
void clearLink(LinkNode* header);
// 尾部添加节点
void append(LinkNode** header, int value);

#endif