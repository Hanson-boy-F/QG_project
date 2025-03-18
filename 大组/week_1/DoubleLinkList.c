#include"DoubleList.h"

//
struct DoubleLinkNode* Linklist()
{
	// 创建头节点
	LinkNode* header = malloc(sizeof(LinkNode));
	header->data = 0;
	header->front = NULL;
	header->next = NULL;

	LinkNode* Next = header;
	int value = 0;
	while (1)
	{
		printf("请为链表输入数据：\n");
		scanf("%d", &value);
		// 当输入0时停止输入
		if (value == 0)
		{
			break;
		}
		// 创建新节点存储输入的数据
		LinkNode* newnode = malloc(sizeof(LinkNode));
		// 赋值
		newnode->data = value;
		newnode->front = Next;
		newnode->next = NULL;
		// 更新指针指向
		Next->next = newnode;
		Next = newnode;

	}
	return header;

}
// 主函数
int main()
{
	LinkNode* header = Linklist();
	
	// 打印链表
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 反转打印链表
	reverseprintLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 插入节点
	insertLink(header,1,666);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 删除指定数据
	deleteLink(&header, 2);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 查找某个数据
	findLink(header, 2);
	printf("\n-------(我是分割线，不必理会)------\n");
	findLink(header, 10000);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 向尾部添加数据
	append(&header, 888);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 判断链表是否为空
	Empty(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 获取链表长度
	Length(header);
	// 清空链表
	clearLink(header);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 判断是否清空
	Empty(header);
	printf("\n-------(我是分割线，不必理会)------\n");




	return 0;
}

// 打印链表
void printLink(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// 辅助指针
	LinkNode* Current = header->next;
	while (Current!=NULL)
	{
		printf("%d ", Current->data);
		// 指针后移
		Current = Current->next;

	}

}

// 反向打印链表
void reverseprintLink(LinkNode* header)
{
	if (header == NULL||header->next==NULL)
	{
		return;
	}
	// 辅助指针
	LinkNode* Current = header->next;
	while (Current->next != NULL)
	{
		// 一直后移
		Current = Current->next;
	}

	while (Current != NULL)
	{
		printf("%d ", Current->data);
		// 指针前移
		Current = Current->front;
	}

}

// 在某个位置插入节点
void insertLink(LinkNode* header, int oldvalue, int newvalue)
{
	if (header == NULL)
	{
		return;
	}
	// 两辅助指针
	LinkNode* Front = header;
	LinkNode* Current = header->next;
	LinkNode* Next = NULL;

	// 创建新节点存放
	LinkNode* newNode = malloc(sizeof(LinkNode));
	newNode->data = newvalue;
	while (Current != NULL)
	{
		// 找到要插入的位置
		if (Current->data == oldvalue)
		{
			break;
		}
		// 更新指针
		Front = Current;
		Current = Current->next;

	}
	newNode->front = Front;
	newNode->next = Current;

	if (Current != NULL)
	{
		Current->front = newNode;
	}
	Front->next = newNode;

}

// 删除指定数据
void deleteLink(LinkNode**header, int deletevalue)
{
	if (header == NULL)
	{
		return;
	}
	// 辅助指针
	LinkNode* Current = *header;

	while (Current != NULL)
	{
		// 能找到要删的数据
		if (Current->data==deletevalue)
		{
			// 更新指针
			if ((Current->front != NULL))
			{
				Current->front->next = Current->next;
			}
			else
			{
				*header = Current->next;
			}
			if (Current->next != NULL)
			{
				Current->next->front = Current->front;
			}
			// 释放内存
			free(Current);
			return;

		}
		// 指针移动
		Current = Current->next;
	}
	// 否则没找到要删的
	printf("Deletevalue is not found.\n");

}

// 查找某个数据
void findLink(LinkNode* header, int value)
{
	if (header == NULL)
	{
		return;
	}
	// 辅助指针
	LinkNode* Current = header->next;
	// 查找
	while (Current != NULL)
	{
		if (Current->data == value)
		{
			printf("I find it.\n");
			// 找到即循环结束
			return;
		}
		// 移动辅助指针
		Current = Current->next;
	}
	printf("It don't at here.\n");

}

// 判断链表是否为空
void Empty(LinkNode* header)
{
	if (header == NULL||header->next==NULL)
	{
		printf("The link is empty.\n");
	}
	else
	{
		printf("The link isn't empty.\n");
	}
}

// 清空链表
void clearLink(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// 辅助指针
	LinkNode* Current = header->next;
	// 利用循环清空链表
	while (Current != NULL)
	{
		// 把数据域干掉，，先保存当前节点的下一个节点地址
		LinkNode* Next = Current->next;
		free(Current);
		// 指针移动
		Current = Next;
	}
	// 指针更新
	header->next = NULL;

}

// 获取链表长度
void Length(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	int len = 0;
	// 辅助指针
	LinkNode* Current = header->next;
	while (Current != NULL)
	{
		len++;
		// 指针移动
		Current = Current->next;
	}
	printf("Link length is %d.\n", len);

}

// 尾部添加节点
void append(LinkNode** header, int value)
{
	// 创建新节点存储输入的数据
	LinkNode* newnode = malloc(sizeof(LinkNode));
	// 赋值
	newnode->data = value;
	newnode->front = NULL;
	newnode->next = NULL;

	if (*header == NULL)
	{
		*header = newnode;
	}
	else
	{
		LinkNode* Current = *header;
		while (Current->next != NULL)
		{
			Current = Current->next;
		}
		// 插入新节点到尾部
		Current->next = newnode;
		newnode->front = Current;

	}

}