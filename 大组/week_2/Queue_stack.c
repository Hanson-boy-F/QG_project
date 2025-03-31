// 栈式队列

#include"Queue.h"


int main()
{
    DataType type = data_type();   // 获取选择的数据类型
    Queue* queue = queue_create(); // 创建队列

    int choice;

    do {
        printf("\n1.录入 2.删除队列数据 3.展示队列数据 4.退出\n> ");
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
                printf("出队: ");
                print_element(data, size, &type);
                printf("\n");
                free(data);   // 释放出队的数据的内存
            }
            else 
            {
                printf("队列为空!\n");
            }
            break;
        }
        case 3:
            printf("队列内容: ");
            // 遍历打印
            queue_foreach(queue, print_element, &type);
            printf("\n");
            break;
        }
    } 
    while (choice != 4);
    {
        // 销毁队列
        queue_destroy(queue);
    }

    return 0;

}

// 获取选择的数据类型
DataType data_type()
{
    int choice;
    printf("请选择要输入数据的数据类型:\n1. 整数\n2. 小数\n3. 字符串\n ");
    scanf("%d", &choice);
    getchar();
    return (DataType)(choice - 1);
}

// 根据选择的数据类型，读取数据
void handle_enqueue(Queue* queue, DataType type)
{
    switch (type) 
    {
    case INT: 
    {
        int value;
        printf("录入整数: ");
        if (scanf("%d", &value) != 1) 
        {
            printf("无效输入!\n");
            // 清空无效输入
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
        printf("录入小数: ");
        if (scanf("%f", &value) != 1) 
        {
            printf("无效输入!\n");
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
        printf("录入字符串: ");
        fgets(buf, sizeof(buf), stdin);
        int len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[--len] = '\0';
        enqueue(queue, buf, len + 1);
        break;
    }
    }

}

// 打印每个元素
void print_element(const void* data, int size, void* user_data) 
{
    DataType type = *(DataType*)user_data;
    switch (type) {
    case INT:    printf("%d ", *(int*)data);     break;
    case FLOAT:  printf("%f ", *(float*)data);   break;
    case STRING: printf("\"%s\" ", (char*)data); break;
    }
}

// 创建
Queue* queue_create()
{
    Queue* queue = (Queue*)malloc(sizeof(Queue));
    queue->front = queue->tail = NULL;
    return queue;
}

// 销毁
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

// 入队
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

// 出队
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

// 检查队列是否为空
int queue_empty(const Queue* queue)
{
    return !queue || !queue->front;
}

// 遍历队列
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