#include"DoubleList.h"

//
struct DoubleLinkNode* Linklist()
{
	// ����ͷ�ڵ�
	LinkNode* header = malloc(sizeof(LinkNode));
	header->data = 0;
	header->front = NULL;
	header->next = NULL;

	LinkNode* Next = header;
	int value = 0;
	while (1)
	{
		printf("��Ϊ�����������ݣ�\n");
		scanf("%d", &value);
		// ������0ʱֹͣ����
		if (value == 0)
		{
			break;
		}
		// �����½ڵ�洢���������
		LinkNode* newnode = malloc(sizeof(LinkNode));
		// ��ֵ
		newnode->data = value;
		newnode->front = Next;
		newnode->next = NULL;
		// ����ָ��ָ��
		Next->next = newnode;
		Next = newnode;

	}
	return header;

}
// ������
int main()
{
	LinkNode* header = Linklist();
	
	// ��ӡ����
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ��ת��ӡ����
	reverseprintLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ����ڵ�
	insertLink(header,1,666);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ɾ��ָ������
	deleteLink(&header, 2);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ����ĳ������
	findLink(header, 2);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	findLink(header, 10000);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ��β���������
	append(&header, 888);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �ж������Ƿ�Ϊ��
	Empty(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// ��ȡ������
	Length(header);
	// �������
	clearLink(header);
	printLink(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");
	// �ж��Ƿ����
	Empty(header);
	printf("\n-------(���Ƿָ��ߣ��������)------\n");




	return 0;
}

// ��ӡ����
void printLink(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// ����ָ��
	LinkNode* Current = header->next;
	while (Current!=NULL)
	{
		printf("%d ", Current->data);
		// ָ�����
		Current = Current->next;

	}

}

// �����ӡ����
void reverseprintLink(LinkNode* header)
{
	if (header == NULL||header->next==NULL)
	{
		return;
	}
	// ����ָ��
	LinkNode* Current = header->next;
	while (Current->next != NULL)
	{
		// һֱ����
		Current = Current->next;
	}

	while (Current != NULL)
	{
		printf("%d ", Current->data);
		// ָ��ǰ��
		Current = Current->front;
	}

}

// ��ĳ��λ�ò���ڵ�
void insertLink(LinkNode* header, int oldvalue, int newvalue)
{
	if (header == NULL)
	{
		return;
	}
	// ������ָ��
	LinkNode* Front = header;
	LinkNode* Current = header->next;
	LinkNode* Next = NULL;

	// �����½ڵ���
	LinkNode* newNode = malloc(sizeof(LinkNode));
	newNode->data = newvalue;
	while (Current != NULL)
	{
		// �ҵ�Ҫ�����λ��
		if (Current->data == oldvalue)
		{
			break;
		}
		// ����ָ��
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

// ɾ��ָ������
void deleteLink(LinkNode**header, int deletevalue)
{
	if (header == NULL)
	{
		return;
	}
	// ����ָ��
	LinkNode* Current = *header;

	while (Current != NULL)
	{
		// ���ҵ�Ҫɾ������
		if (Current->data==deletevalue)
		{
			// ����ָ��
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
			// �ͷ��ڴ�
			free(Current);
			return;

		}
		// ָ���ƶ�
		Current = Current->next;
	}
	// ����û�ҵ�Ҫɾ��
	printf("Deletevalue is not found.\n");

}

// ����ĳ������
void findLink(LinkNode* header, int value)
{
	if (header == NULL)
	{
		return;
	}
	// ����ָ��
	LinkNode* Current = header->next;
	// ����
	while (Current != NULL)
	{
		if (Current->data == value)
		{
			printf("I find it.\n");
			// �ҵ���ѭ������
			return;
		}
		// �ƶ�����ָ��
		Current = Current->next;
	}
	printf("It don't at here.\n");

}

// �ж������Ƿ�Ϊ��
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

// �������
void clearLink(LinkNode* header)
{
	if (header == NULL)
	{
		return;
	}
	// ����ָ��
	LinkNode* Current = header->next;
	// ����ѭ���������
	while (Current != NULL)
	{
		// ��������ɵ������ȱ��浱ǰ�ڵ����һ���ڵ��ַ
		LinkNode* Next = Current->next;
		free(Current);
		// ָ���ƶ�
		Current = Next;
	}
	// ָ�����
	header->next = NULL;

}

// ��ȡ������
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
		// ָ���ƶ�
		Current = Current->next;
	}
	printf("Link length is %d.\n", len);

}

// β����ӽڵ�
void append(LinkNode** header, int value)
{
	// �����½ڵ�洢���������
	LinkNode* newnode = malloc(sizeof(LinkNode));
	// ��ֵ
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
		// �����½ڵ㵽β��
		Current->next = newnode;
		newnode->front = Current;

	}

}