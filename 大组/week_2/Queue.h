#define _CRT_SECURE_NO_WARNINGS

#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef QUEUE
#define QUEUE

// 声明结构体 Queue和QueueNode
typedef struct QueueNode
{
	void* data;			//数据指针，void指针支持多种数据类型
	int data_size;   // 队列大小
	struct QueueNode* next;   // 指向下一个结点的指针
}QueueNode;
typedef struct Queue
{
	QueueNode* front;
	QueueNode* tail;
}Queue;

// 定义一个返回值类型是void的函数指针类型QueueCallback,回调函数，用于遍历队列时调用每个元素
typedef void (*QueueCallback)(const void* data, int data_size, void* user_data);
// 枚举-数据类型
typedef enum DataType { INT, FLOAT, STRING } DataType;

// 创建
Queue* queue_create();
// 销毁
void queue_destroy(Queue* queue);
// 入队 enter queue
void enqueue(Queue* queue, const void* data, int data_size);
// 出队 delete queue
void* dequeue(Queue* queue, int* data_size);
// 检查是否为空
int queue_empty(const Queue* queue);
// 遍历队列
void queue_foreach(const Queue* queue, QueueCallback callback, void* user_data);

// 根据选择的数据类型，读取数据
 void handle_enqueue(Queue* queue, DataType type);
// 打印每个元素
void print_element(const void* data, int size, void* user_data);
// 获取选择的数据类型
DataType data_type();

#endif
