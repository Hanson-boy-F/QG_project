# include"SingleList.h"

struct SingleListNode* LinkList()
{
	// 创建头结点
	LinkNode* header = malloc(sizeof(LinkNode));
	header->data = 0;
	header->next = NULL;

	// 尾部指针始终指向尾部
	LinkNode* Next = header;
	// 通过循环为链表输入数据
	int value = 0;
	// 无限循环
	while (1)
	{
		printf("请为链表输入数据：\n");
		scanf("%d", &value);
		// 无限循环结束条件--当输入0
		if (value == 0)
		{
			break;
		}
		// 创建新节点存储输入的数据
		LinkNode* newnode = malloc(sizeof(LinkNode));
		// 输入的数据赋给新节点
		newnode->data = value;
		newnode->next = NULL;
		// 更新指针指向
		Next->next =newnode;
		Next = newnode;

	}
	return header;

}
// 主函数
int main()
{
	LinkNode* header = LinkList();

	// 打印链表
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 能找到oldvalue
	append(header, 2, 666);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 不能找到oldvalue
	append(header, 10, 666);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 奇偶互换
	LinkNode*oddEvenSwap(LinkNode * header);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 找到链表中点
	LinkNode*middle= findMiddle(header);
	printf("Middle is %d\n", middle?middle->data:-1);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 反向打印
	printReverse(header->next);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 判断链表是不是环
	Cycle(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 消除某个值
	Remove(header, 2);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 找某个值--2   消去了
	Findvalue(header, 2);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 找某个值--1
	Findvalue(header, 1);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 判断链表是不是空
	Empty(header);
	// 清空链表
	CLearLink(header);
	printLink(header);
	printf("\n-------(我是分割线，不必理会)------\n");
	// 判断链表是不是空
	Empty(header);


	system("pause");

	return 0;
}

// 打印链表
void printLink(LinkNode* header)
{
	if ( header==NULL)
	{
		return;
	}
	// 辅助指针变量
	LinkNode* Current = header->next;
	while (Current != NULL)
	{
		printf("%d ", Current->data);
		// 指针后移
		Current = Current->next;
	}

}

// 在某位置插入新数据
void append(LinkNode* header, int oldvalue, int newvalue)
{
	if (header ==NULL)
	{
		return;
	}
	// 需要两个辅助指针,一起后移
	LinkNode* Front = header;
	LinkNode* Current = Front->next;

	while (Current != NULL)
	{
		// 找到插入的位置,循环结束，否则辅助指针后移继续找
		if (Current->data == oldvalue)
		{
			break;
		}
		// 没找到，后移
		Front = Current;
		Current = Current->next;
	}
	// 如果Current为空，即不存在oldvalue，即不插入
	if(Current==NULL)
	{
		printf("未找到%d，无法插入%d\n", oldvalue, newvalue);
		return;
	}
	// 创建新节点存放要插入的数据
	LinkNode* newnode = malloc(sizeof(LinkNode));
	// 赋值
	newnode->data = newvalue;
	newnode->next = NULL;
	// 辅助指针更新
	Front->next = newnode;
	newnode->next = Current;
}

// 反向遍历
void printReverse(LinkNode* header) 
{
	if (header == NULL) 
	{
		return;
	}
	// 先递归到最后的节点
	printReverse(header->next);
	// 打印当前节点的数据
	printf("%d ", header->data);
}

// 删除特定值
void Remove(LinkNode* header, int deleteValue)
{
	if (header == NULL)
	{
		return;
	}
	// 需要两个辅助指针
	LinkNode* Front = header;
	LinkNode* Current = header->next;

	while (Current != NULL)
	{
		// 找到要删的值为止
		if (Current->data == deleteValue)
		{
			break;
		}
		// 没找到，移动两指针
		Front = Current;
		Current = Current->next;

	}
	// 如果遍历完都没找到
	if (Current == NULL)
	{
		return;
	}
	// 找到
	else
	{
		// 更新指针
		Front->next = Current->next;
		// 释放删除节点的内存
		free(Current);
		Current = NULL;
	}

}

// 清空链表
void CLearLink(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// 辅助指针
	LinkNode* Current = header->next;     // 跳过头字节，即保留头字节

	while (Current != NULL)
	{
		// 将数据域删除，且先保存下一个节点地址
		LinkNode* Next = Current->next;
		// 释放空间
		free(Current);
		// 指向下一个
		Current = Next;
	}
	// 最后释放空间
	header->next = NULL;
}

// 查找某数据是否存在
void Findvalue(LinkNode* header, int findValue)
{
	if (header == NULL)
	{
		return;
	}

	// 辅助指针
	LinkNode* Current = header->next;  // 从第一个实际节点开始
	int found = 0;  // 用于标记是否找到目标值

	while (Current != NULL)
	{
		if (Current->data == findValue)
		{
			printf("I find it!\n");
			found = 1;  // 标记为已找到
			break;      // 找到后退出循环
		}
		Current = Current->next;  // 指向下一个节点
	}

	// 如果循环结束，且没有找到目标值
	if (!found)
	{
		printf("It don't at here.\n");
	}
}

// 奇偶互换
LinkNode* oddEvenSwap(LinkNode* header) 
{
	if (header == NULL || header->next == NULL) 
	{
		return header; 
		// 如果链表为空或只有一个节点，直接返回
	}
	// 奇数节点指针
	LinkNode* odd = header; 
	// 偶数节点指针
	LinkNode* even = header->next;
	// 保存偶数头，以便最后连接
	LinkNode* evenHead = even; 

	while (even != NULL && even->next != NULL) 
	{
		// 将奇数节点指向下一个奇数节点
		odd->next = even->next; 
		// 移动至下一个奇数节点
		odd = odd->next; 
		// 将偶数节点指向下一个偶数节点
		even->next = odd->next; 
		// 移动至下一个偶数节点
		even = even->next; 
	}
	// 将最后一个奇数节点指向偶数头
	odd->next = evenHead; 
	// 返回新的头节点
	return header; 
}

// 找到链表中点
LinkNode* findMiddle(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// 两个辅助指针
	LinkNode* slow = header;
	LinkNode* fast = header;

	while (fast != NULL && fast->next != NULL)
	{
		slow = slow->next;
		fast = fast->next->next;
	}

	return slow;
}

// 判断链表是否成环
void Cycle(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// 两个辅助指针
	LinkNode* slow = header;
	LinkNode* fast = header;
	while (fast != NULL && fast->next != NULL)
	{
		slow = slow->next;
		fast = fast->next->next;
		if (slow == fast)
		{
			printf("It is cycle.\n");
			return;
		}
		else
		{
			printf("It isn't cycle.\n");
			return;
		}

	}

}

// 获取链表的长度
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
		Current = Current -> next;
	}
	printf("Link length is %d\n", len);
}

// 判断链表是不是空链表
void Empty(LinkNode* header)
{
	if (header == NULL || header->next == NULL)
	{
		printf("It is empty");
		return;
	}
	else
	{
		printf("It isn't empty.");
	}

}
