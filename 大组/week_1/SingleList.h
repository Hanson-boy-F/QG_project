#define _CRT_SECURE_NO_WARNINGS

#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>

#ifndef SingleList
// ����ڵ�����
typedef struct SingleListNode
{
	int data;
	struct SingleListNode* next;

}LinkNode;
// ��ʼ������
struct SingleListNode* LinkList();
// ��ӡ����
void printLink(LinkNode* header);
// ����������
void append(LinkNode* header, int oldvalue, int newvalue);
// �������
void printReverse(LinkNode* node);
// ɾ���ض�ֵ
void Remove(LinkNode* header, int deleteValue);
// �������
void CLearLink(LinkNode* header);
// ����ĳֵ
void Findvalue(LinkNode* header, int findValue);
// �ǵݹ鷴ת
void Norecursion(LinkNode* header);
// ��ż��ת
LinkNode* oddEvenSwap(LinkNode* header);
// �ҵ������е�
LinkNode* findMiddle(LinkNode* head);
// �ж������ǲ��ǻ�
void Cycle(LinkNode* header);
// ��ȡ����ĳ���
void Length(LinkNode* header);
// �ж������ǲ��ǿ�����
void Empty(LinkNode* header);

#endif