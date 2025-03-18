# include"SingleList.h"

struct SingleListNode* LinkList()
{
	// ����ͷ���
	LinkNode* header = malloc(sizeof(LinkNode));
	header->data = 0;
	header->next = NULL;

	// β��ָ��ʼ��ָ��β��
	LinkNode* Next = header;
	// ͨ��ѭ��Ϊ������������
	int value = 0;
	// ����ѭ��
	while (1)
	{
		printf("��Ϊ�����������ݣ�\n");
		scanf("%d", &value);
		// ����ѭ����������--������0
		if (value == 0)
		{
			break;
		}
		// �����½ڵ�洢���������
		LinkNode* newnode = malloc(sizeof(LinkNode));
		// ��������ݸ����½ڵ�
		newnode->data = value;
		newnode->next = NULL;
		// ����ָ��ָ��
		Next->next =newnode;
		Next = newnode;

	}
	return header;

}
// ������
int main()
{
	LinkNode* header = LinkList();

	// ��ӡ����
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ���ҵ�oldvalue
	append(header, 2, 666);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �����ҵ�oldvalue
	append(header, 10, 666);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ��ż����
	LinkNode*oddEvenSwap(LinkNode * header);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �ҵ������е�
	LinkNode*middle= findMiddle(header);
	printf("Middle is %d\n", middle?middle->data:-1);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �����ӡ
	printReverse(header->next);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �ж������ǲ��ǻ�
	Cycle(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ����ĳ��ֵ
	Remove(header, 2);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ��ĳ��ֵ--2   ��ȥ��
	Findvalue(header, 2);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ��ĳ��ֵ--1
	Findvalue(header, 1);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �ж������ǲ��ǿ�
	Empty(header);
	// �������
	CLearLink(header);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �ж������ǲ��ǿ�
	Empty(header);


	system("pause");

	return 0;
}

// ��ӡ����
void printLink(LinkNode* header)
{
	if ( header==NULL)
	{
		return;
	}
	// ����ָ�����
	LinkNode* Current = header->next;
	while (Current != NULL)
	{
		printf("%d ", Current->data);
		// ָ�����
		Current = Current->next;
	}

}

// ��ĳλ�ò���������
void append(LinkNode* header, int oldvalue, int newvalue)
{
	if (header ==NULL)
	{
		return;
	}
	// ��Ҫ��������ָ��,һ�����
	LinkNode* Front = header;
	LinkNode* Current = Front->next;

	while (Current != NULL)
	{
		// �ҵ������λ��,ѭ��������������ָ����Ƽ�����
		if (Current->data == oldvalue)
		{
			break;
		}
		// û�ҵ�������
		Front = Current;
		Current = Current->next;
	}
	// ���CurrentΪ�գ���������oldvalue����������
	if(Current==NULL)
	{
		printf("δ�ҵ�%d���޷�����%d\n", oldvalue, newvalue);
		return;
	}
	// �����½ڵ���Ҫ���������
	LinkNode* newnode = malloc(sizeof(LinkNode));
	// ��ֵ
	newnode->data = newvalue;
	newnode->next = NULL;
	// ����ָ�����
	Front->next = newnode;
	newnode->next = Current;
}

// �������
void printReverse(LinkNode* header) 
{
	if (header == NULL) 
	{
		return;
	}
	// �ȵݹ鵽���Ľڵ�
	printReverse(header->next);
	// ��ӡ��ǰ�ڵ������
	printf("%d ", header->data);
}

// ɾ���ض�ֵ
void Remove(LinkNode* header, int deleteValue)
{
	if (header == NULL)
	{
		return;
	}
	// ��Ҫ��������ָ��
	LinkNode* Front = header;
	LinkNode* Current = header->next;

	while (Current != NULL)
	{
		// �ҵ�Ҫɾ��ֵΪֹ
		if (Current->data == deleteValue)
		{
			break;
		}
		// û�ҵ����ƶ���ָ��
		Front = Current;
		Current = Current->next;

	}
	// ��������궼û�ҵ�
	if (Current == NULL)
	{
		return;
	}
	// �ҵ�
	else
	{
		// ����ָ��
		Front->next = Current->next;
		// �ͷ�ɾ���ڵ���ڴ�
		free(Current);
		Current = NULL;
	}

}

// �������
void CLearLink(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// ����ָ��
	LinkNode* Current = header->next;     // ����ͷ�ֽڣ�������ͷ�ֽ�

	while (Current != NULL)
	{
		// ��������ɾ�������ȱ�����һ���ڵ��ַ
		LinkNode* Next = Current->next;
		// �ͷſռ�
		free(Current);
		// ָ����һ��
		Current = Next;
	}
	// ����ͷſռ�
	header->next = NULL;
}

// ����ĳ�����Ƿ����
void Findvalue(LinkNode* header, int findValue)
{
	if (header == NULL)
	{
		return;
	}

	// ����ָ��
	LinkNode* Current = header->next;  // �ӵ�һ��ʵ�ʽڵ㿪ʼ
	int found = 0;  // ���ڱ���Ƿ��ҵ�Ŀ��ֵ

	while (Current != NULL)
	{
		if (Current->data == findValue)
		{
			printf("I find it!\n");
			found = 1;  // ���Ϊ���ҵ�
			break;      // �ҵ����˳�ѭ��
		}
		Current = Current->next;  // ָ����һ���ڵ�
	}

	// ���ѭ����������û���ҵ�Ŀ��ֵ
	if (!found)
	{
		printf("It don't at here.\n");
	}
}

// ��ż����
LinkNode* oddEvenSwap(LinkNode* header) 
{
	if (header == NULL || header->next == NULL) 
	{
		return header; 
		// �������Ϊ�ջ�ֻ��һ���ڵ㣬ֱ�ӷ���
	}
	// �����ڵ�ָ��
	LinkNode* odd = header; 
	// ż���ڵ�ָ��
	LinkNode* even = header->next;
	// ����ż��ͷ���Ա��������
	LinkNode* evenHead = even; 

	while (even != NULL && even->next != NULL) 
	{
		// �������ڵ�ָ����һ�������ڵ�
		odd->next = even->next; 
		// �ƶ�����һ�������ڵ�
		odd = odd->next; 
		// ��ż���ڵ�ָ����һ��ż���ڵ�
		even->next = odd->next; 
		// �ƶ�����һ��ż���ڵ�
		even = even->next; 
	}
	// �����һ�������ڵ�ָ��ż��ͷ
	odd->next = evenHead; 
	// �����µ�ͷ�ڵ�
	return header; 
}

// �ҵ������е�
LinkNode* findMiddle(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// ��������ָ��
	LinkNode* slow = header;
	LinkNode* fast = header;

	while (fast != NULL && fast->next != NULL)
	{
		slow = slow->next;
		fast = fast->next->next;
	}

	return slow;
}

// �ж������Ƿ�ɻ�
void Cycle(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// ��������ָ��
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

// ��ȡ����ĳ���
void Length(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	int len = 0;
	// ����ָ��
	LinkNode* Current = header->next;
	while (Current != NULL)
	{
		len++;
		Current = Current -> next;
	}
	printf("Link length is %d\n", len);
}

// �ж������ǲ��ǿ�����
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
