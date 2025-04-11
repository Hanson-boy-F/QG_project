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

        // 安全读取选择-防御代码-确保健壮
        if (!safeInputInt("", &choice)) 
        {
            printf("无效输入，请输入数字\n");
            while (getchar() != '\n'); // 清空输入缓冲区
            continue;
        }

        switch (choice) 
        {
        case 0: // 退出
            freeTree(tree.root);
            printf("程序已退出\n");
            return 0;

        case 1: // 插入
            if (safeInputInt("请输入要插入的整数值: ", &value)) 
            {
                result = BST_insert(&tree, value);
                printf("%s\n", result ? "插入成功" : "插入失败（可能已存在）");
            }
            break;

        case 2: // 删除
            if (safeInputInt("请输入要删除的整数值: ", &value)) 
            {
                result = BST_delete(&tree, value);
                printf("%s\n", result ? "删除成功" : "删除失败（节点不存在）");
            }
            break;

        case 3: // 查找
            if (safeInputInt("请输入要查找的整数值: ", &value)) 
            {
                result = BST_search(&tree, value);
                printf("%s\n", result ? "存在" : "不存在");
            }
            break;

        case 4: // 前序递归
            printf("前序遍历(递归): ");
            BST_preorderR(&tree, printNode);
            printf("\n");
            break;

        case 5: // 中序递归
            printf("中序遍历(递归): ");
            BST_inorderR(&tree, printNode);
            printf("\n");
            break;

        case 6: // 后序递归
            printf("后序遍历(递归): ");
            BST_postorderR(&tree, printNode);
            printf("\n");
            break;

        case 7: // 前序非递归
            printf("前序遍历(非递归): ");
            BST_preorderI(&tree, printNode);
            printf("\n");
            break;

        case 8: // 中序非递归
            printf("中序遍历(非递归): ");
            BST_inorderI(&tree, printNode);
            printf("\n");
            break;

        case 9: // 后序非递归
            printf("后序遍历(非递归): ");
            BST_postorderI(&tree, printNode);
            printf("\n");
            break;

        case 10: // 层序
            printf("层序遍历: ");
            BST_levelOrder(&tree, printNode);
            printf("\n");
            break;

        default:
            printf("无效选择，请重新输入\n");
            break;
        }

        // 清空输入缓冲区
        while (getchar() != '\n');
    }

    return 0;
}


// 安全输入函数
int safeInputInt(const char* prompt, int* value)
{
    char buffer[500];  // 缓存区
    char* endptr;

    while (1)
    {
        printf("%s", prompt);
        if (!fgets(buffer, sizeof(buffer), stdin))
        {
            return failed; // 输入错误
        }

        // 尝试转换整数
        *value = (int)strtol(buffer, &endptr, 10);

        // 检查整个输入是否有效
        if (endptr == buffer || (*endptr != '\n' && *endptr != '\0'))
        {
            printf("无效输入，请重新输入整数\n");
            continue;
        }
        return succeed;
    }
}

// 打印节点值
void printNode(NodePtr node)
{
    if (node) printf("%d ", node->value);
}

// 释放树内存
void freeTree(NodePtr node)
{
    if (node)
    {
        freeTree(node->left);
        freeTree(node->right);
        free(node);
    }
}

// 显示交互菜单
void showMenu()
{
    printf("\n=== 二叉排序树操作菜单 ===\n");
    printf("1. 插入节点\n");
    printf("2. 删除节点\n");
    printf("3. 查找节点\n");
    printf("4. 前序遍历(递归)\n");
    printf("5. 中序遍历(递归)\n");
    printf("6. 后序遍历(递归)\n");
    printf("7. 前序遍历(非递归)\n");
    printf("8. 中序遍历(非递归)\n");
    printf("9. 后序遍历(非递归)\n");
    printf("10. 层序遍历\n");
    printf("0. 退出\n");
    printf("请输入选择: ");
}