#define _CRT_SECURE_NO_WARNINGS

#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef QUEUE
#define QUEUE

// �����ṹ�� Queue��QueueNode
typedef struct QueueNode
{
	void* data;			//����ָ�룬voidָ��֧�ֶ�����������
	int data_size;   // ���д�С
	struct QueueNode* next;   // ָ����һ������ָ��
}QueueNode;
typedef struct Queue
{
	QueueNode* front;
	QueueNode* tail;
}Queue;

// ����һ������ֵ������void�ĺ���ָ������QueueCallback,�ص����������ڱ�������ʱ����ÿ��Ԫ��
typedef void (*QueueCallback)(const void* data, int data_size, void* user_data);
// ö��-��������
typedef enum DataType { INT, FLOAT, STRING } DataType;

// ����
Queue* queue_create();
// ����
void queue_destroy(Queue* queue);
// ��� enter queue
void enqueue(Queue* queue, const void* data, int data_size);
// ���� delete queue
void* dequeue(Queue* queue, int* data_size);
// ����Ƿ�Ϊ��
int queue_empty(const Queue* queue);
// ��������
void queue_foreach(const Queue* queue, QueueCallback callback, void* user_data);

// ����ѡ����������ͣ���ȡ����
 void handle_enqueue(Queue* queue, DataType type);
// ��ӡÿ��Ԫ��
void print_element(const void* data, int size, void* user_data);
// ��ȡѡ�����������
DataType data_type();

#endif
