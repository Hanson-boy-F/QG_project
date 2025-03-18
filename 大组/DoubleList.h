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
// ��ʼ������
struct DoubleLinkNode* Linklist();
// ��ӡ����
void printLink(LinkNode* header);
// �����ӡ����
void reverseprintLink(LinkNode* header);
// ��ָ��λ�ò���ڵ�
void insertLink(LinkNode* header, int oldvalue, int newvalue);
// ɾ��ָ������
void deleteLink(LinkNode** header, int deletevalue);
// ����ĳ������
void findLink(LinkNode* header, int value);
// �ж������Ƿ�Ϊ��
void Empty(LinkNode* header);
// ��ȡ������
void Length(LinkNode* header);
// �������
void clearLink(LinkNode* header);
// β����ӽڵ�
void append(LinkNode** header, int value);

#endif