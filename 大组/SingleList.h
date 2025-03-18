#define _CRT_SECURE_NO_WARNINGS

#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>

#ifndef SingleList
// 定义节点类型
typedef struct SingleListNode
{
	int data;
	struct SingleListNode* next;

}LinkNode;
// 初始化链表
struct SingleListNode* LinkList();
// 打印链表
void printLink(LinkNode* header);
// 插入新数据
void append(LinkNode* header, int oldvalue, int newvalue);
// 反向遍历
void printReverse(LinkNode* node);
// 删除特定值
void Remove(LinkNode* header, int deleteValue);
// 清空链表
void CLearLink(LinkNode* header);
// 查找某值
void Findvalue(LinkNode* header, int findValue);
// 非递归反转
void Norecursion(LinkNode* header);
// 奇偶反转
LinkNode* oddEvenSwap(LinkNode* header);
// 找到链表中点
LinkNode* findMiddle(LinkNode* head);
// 判断链表是不是环
void Cycle(LinkNode* header);
// 获取链表的长度
void Length(LinkNode* header);
// 判断链表是不是空链表
void Empty(LinkNode* header);

#endif