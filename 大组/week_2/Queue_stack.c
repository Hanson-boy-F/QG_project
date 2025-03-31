// ջʽ����

#include"Queue.h"


int main()
{
    DataType type = data_type();   // ��ȡѡ�����������
    Queue* queue = queue_create(); // ��������

    int choice;

    do {
        printf("\n1.¼�� 2.ɾ���������� 3.չʾ�������� 4.�˳�\n> ");
        scanf("%d", &choice);

        getchar();

        switch (choice) 
        {
        case 1:
            handle_enqueue(queue, type);
            break;
        case 2: 
            {
            int size;
            void* data = dequeue(queue, &size);
            if (data) 
            {
                printf("����: ");
                print_element(data, size, &type);
                printf("\n");
                free(data);   // �ͷų��ӵ����ݵ��ڴ�
            }
            else 
            {
                printf("����Ϊ��!\n");
            }
            break;
        }
        case 3:
            printf("��������: ");
            // ������ӡ
            queue_foreach(queue, print_element, &type);
            printf("\n");
            break;
        }
    } 
    while (choice != 4);
    {
        // ���ٶ���
        queue_destroy(queue);
    }

    return 0;

}

// ��ȡѡ�����������
DataType data_type()
{
    int choice;
    printf("��ѡ��Ҫ�������ݵ���������:\n1. ����\n2. С��\n3. �ַ���\n ");
    scanf("%d", &choice);
    getchar();
    return (DataType)(choice - 1);
}

// ����ѡ����������ͣ���ȡ����
void handle_enqueue(Queue* queue, DataType type)
{
    switch (type) 
    {
    case INT: 
    {
        int value;
        printf("¼������: ");
        if (scanf("%d", &value) != 1) 
        {
            printf("��Ч����!\n");
            // �����Ч����
            while (getchar() != '\n');
            return;
        }
        getchar();  
        enqueue(queue, &value, sizeof(int));
        break;
    }
    case FLOAT: 
    {
        float value;
        printf("¼��С��: ");
        if (scanf("%f", &value) != 1) 
        {
            printf("��Ч����!\n");
            while (getchar() != '\n');
            return;
        }
        getchar();
        enqueue(queue, &value, sizeof(float));
        break;
    }
    case STRING: 
    {
        char buf[300];
        printf("¼���ַ���: ");
        fgets(buf, sizeof(buf), stdin);
        int len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[--len] = '\0';
        enqueue(queue, buf, len + 1);
        break;
    }
    }

}

// ��ӡÿ��Ԫ��
void print_element(const void* data, int size, void* user_data) 
{
    DataType type = *(DataType*)user_data;
    switch (type) {
    case INT:    printf("%d ", *(int*)data);     break;
    case FLOAT:  printf("%f ", *(float*)data);   break;
    case STRING: printf("\"%s\" ", (char*)data); break;
    }
}

// ����
Queue* queue_create()
{
    Queue* queue = (Queue*)malloc(sizeof(Queue));
    queue->front = queue->tail = NULL;
    return queue;
}

// ����
void queue_destroy(Queue* queue) 
{
    if (!queue) return;

    QueueNode* current = queue->front;
    while (current) 
    {
        QueueNode* next = current->next;
        free(current->data);
        free(current);
        current = next;
    }
    free(queue);
}

// ���
void enqueue(Queue* queue, const void* data, int data_size) 
{
    if (!queue || !data || data_size == 0)
    {
        return;
    }
    QueueNode* new_node = (QueueNode*)malloc(sizeof(QueueNode));
    new_node->data = malloc(data_size);
    memcpy(new_node->data, data, data_size);
    new_node->data_size = data_size;
    new_node->next = NULL;

    if (!queue->tail) 
    {
        queue->front = queue->tail = new_node;
    }
    else 
    {
        queue->tail->next = new_node;
        queue->tail = new_node;
    }
}

// ����
void* dequeue(Queue* queue, int* data_size) 
{
    if (!queue || !queue->front) 
    {
        *data_size = 0;
        return NULL;
    }

    QueueNode* temp = queue->front;
    void* data = malloc(temp->data_size);
    memcpy(data, temp->data, temp->data_size);
    *data_size = temp->data_size;

    queue->front = queue->front->next;
    if (!queue->front) queue->tail = NULL;

    free(temp->data);
    free(temp);
    return data;
}

// �������Ƿ�Ϊ��
int queue_empty(const Queue* queue)
{
    return !queue || !queue->front;
}

// ��������
void queue_foreach(const Queue* queue, QueueCallback callback, void* user_data) 
{
    if (!queue) return;

    QueueNode* current = queue->front;
    while (current)
    {
        callback(current->data, current->data_size, user_data);
        current = current->next;
    }
}