// src/main.c 
#include "binary_sort_tree.h"


int safeInputInt(const char* prompt, int* value);
void printNode(NodePtr node);
void freeTree(NodePtr node);
void showMenu();

int main() 
{
    BinarySortTree tree;
    BST_init(&tree);
    int choice, value;
    Status result;

    while (1) 
    {
        showMenu();

        // ��ȫ��ȡѡ��-��������-ȷ����׳
        if (!safeInputInt("", &choice)) 
        {
            printf("��Ч���룬����������\n");
            while (getchar() != '\n'); // ������뻺����
            continue;
        }

        switch (choice) 
        {
        case 0: // �˳�
            freeTree(tree.root);
            printf("�������˳�\n");
            return 0;

        case 1: // ����
            if (safeInputInt("������Ҫ���������ֵ: ", &value)) 
            {
                result = BST_insert(&tree, value);
                printf("%s\n", result ? "����ɹ�" : "����ʧ�ܣ������Ѵ��ڣ�");
            }
            break;

        case 2: // ɾ��
            if (safeInputInt("������Ҫɾ��������ֵ: ", &value)) 
            {
                result = BST_delete(&tree, value);
                printf("%s\n", result ? "ɾ���ɹ�" : "ɾ��ʧ�ܣ��ڵ㲻���ڣ�");
            }
            break;

        case 3: // ����
            if (safeInputInt("������Ҫ���ҵ�����ֵ: ", &value)) 
            {
                result = BST_search(&tree, value);
                printf("%s\n", result ? "����" : "������");
            }
            break;

        case 4: // ǰ��ݹ�
            printf("ǰ�����(�ݹ�): ");
            BST_preorderR(&tree, printNode);
            printf("\n");
            break;

        case 5: // ����ݹ�
            printf("�������(�ݹ�): ");
            BST_inorderR(&tree, printNode);
            printf("\n");
            break;

        case 6: // ����ݹ�
            printf("�������(�ݹ�): ");
            BST_postorderR(&tree, printNode);
            printf("\n");
            break;

        case 7: // ǰ��ǵݹ�
            printf("ǰ�����(�ǵݹ�): ");
            BST_preorderI(&tree, printNode);
            printf("\n");
            break;

        case 8: // ����ǵݹ�
            printf("�������(�ǵݹ�): ");
            BST_inorderI(&tree, printNode);
            printf("\n");
            break;

        case 9: // ����ǵݹ�
            printf("�������(�ǵݹ�): ");
            BST_postorderI(&tree, printNode);
            printf("\n");
            break;

        case 10: // ����
            printf("�������: ");
            BST_levelOrder(&tree, printNode);
            printf("\n");
            break;

        default:
            printf("��Чѡ������������\n");
            break;
        }

        // ������뻺����
        while (getchar() != '\n');
    }

    return 0;
}


// ��ȫ���뺯��
int safeInputInt(const char* prompt, int* value)
{
    char buffer[500];  // ������
    char* endptr;

    while (1)
    {
        printf("%s", prompt);
        if (!fgets(buffer, sizeof(buffer), stdin))
        {
            return failed; // �������
        }

        // ����ת������
        *value = (int)strtol(buffer, &endptr, 10);

        // ������������Ƿ���Ч
        if (endptr == buffer || (*endptr != '\n' && *endptr != '\0'))
        {
            printf("��Ч���룬��������������\n");
            continue;
        }
        return succeed;
    }
}

// ��ӡ�ڵ�ֵ
void printNode(NodePtr node)
{
    if (node) printf("%d ", node->value);
}

// �ͷ����ڴ�
void freeTree(NodePtr node)
{
    if (node)
    {
        freeTree(node->left);
        freeTree(node->right);
        free(node);
    }
}

// ��ʾ�����˵�
void showMenu()
{
    printf("\n=== ���������������˵� ===\n");
    printf("1. ����ڵ�\n");
    printf("2. ɾ���ڵ�\n");
    printf("3. ���ҽڵ�\n");
    printf("4. ǰ�����(�ݹ�)\n");
    printf("5. �������(�ݹ�)\n");
    printf("6. �������(�ݹ�)\n");
    printf("7. ǰ�����(�ǵݹ�)\n");
    printf("8. �������(�ǵݹ�)\n");
    printf("9. �������(�ǵݹ�)\n");
    printf("10. �������\n");
    printf("0. �˳�\n");
    printf("������ѡ��: ");
}