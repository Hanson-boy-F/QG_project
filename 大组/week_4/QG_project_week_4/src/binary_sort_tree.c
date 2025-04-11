// src/binary_sort_tree.c
#include "binary_sort_tree.h"
#include <stdio.h>
#include <stdlib.h>

// 初始化二叉排序树
Status BST_init(BinarySortTreePtr tree) 
{
    if (tree == NULL) 
        return failed;
    tree->root = NULL;
    return succeed;
}

// 创建新节点
static NodePtr createNode(ElemType value) 
{
    NodePtr newNode = (NodePtr)malloc(sizeof(Node));
    if (newNode) 
    {
        newNode->value = value;
        newNode->left = newNode->right = NULL;
    }
    return newNode;
}

// 插入节点（内部辅助函数）
static Status insertNode(NodePtr* root, ElemType value) 
{
    if (*root == NULL) 
    {
        *root = createNode(value);
        return *root ? succeed : failed;
    }
    if (value < (*root)->value)
        return insertNode(&(*root)->left, value);
    else if (value > (*root)->value)
        return insertNode(&(*root)->right, value);
    return failed; // 值已存在
}

Status BST_insert(BinarySortTreePtr tree, ElemType value) 
{
    return tree ? insertNode(&tree->root, value) : failed;
}

// 删除节点（内部辅助函数）
static NodePtr minValueNode(NodePtr node) 
{
    while (node->left != NULL)
        node = node->left;
    return node;
}

static NodePtr deleteNode(NodePtr root, ElemType value, Status* success) {
    if (root == NULL) 
        return root;

    if (value < root->value) 
    {
        root->left = deleteNode(root->left, value, success);
    }
    else if (value > root->value) 
    {
        root->right = deleteNode(root->right, value, success);
    }
    else 
    {
        *success = succeed;
        if (root->left == NULL) 
        {
            NodePtr temp = root->right;
            free(root);
            return temp;
        }
        else if (root->right == NULL) 
        {
            NodePtr temp = root->left;
            free(root);
            return temp;
        }

        NodePtr temp = minValueNode(root->right);
        root->value = temp->value;
        root->right = deleteNode(root->right, temp->value, success);
    }
    return root;
}

Status BST_delete(BinarySortTreePtr tree, ElemType value) 
{
    Status success = failed;
    if (tree) 
    {
        tree->root = deleteNode(tree->root, value, &success);
    }
    return success;
}

// 查找节点
Status BST_search(BinarySortTreePtr tree, ElemType value) 
{
    NodePtr current = tree->root;
    while (current) 
    {
        if (value == current->value) return true;
        current = value < current->value ? current->left : current->right;
    }
    return false;
}

// 遍历辅助数据结构
typedef struct 
{
    NodePtr* data;
    int top;
    int capacity;
} Stack;

static Stack* createStack(int capacity) 
{
    Stack* stack = (Stack*)malloc(sizeof(Stack));
    stack->data = (NodePtr*)malloc(capacity * sizeof(NodePtr));
    stack->top = -1;
    stack->capacity = capacity;
    return stack;
}

static void push(Stack* stack, NodePtr node) 
{
    if (stack->top < stack->capacity - 1)
        stack->data[++stack->top] = node;
}

static NodePtr pop(Stack* stack) 
{
    return stack->top >= 0 ? stack->data[stack->top--] : NULL;
}

// 非递归前序遍历
Status BST_preorderI(BinarySortTreePtr tree, void (*visit)(NodePtr)) 
{
    if (!tree || !tree->root) return failed;

    Stack* stack = createStack(100);
    push(stack, tree->root);

    while (stack->top != -1) 
    {
        NodePtr node = pop(stack);
        visit(node);
        if (node->right) push(stack, node->right);
        if (node->left) push(stack, node->left);
    }

    free(stack->data);
    free(stack);
    return succeed;
}

// 递归遍历辅助函数
static void preorder(NodePtr node, void (*visit)(NodePtr)) 
{
    if (node) {
        visit(node);
        preorder(node->left, visit);
        preorder(node->right, visit);
    }
}

Status BST_preorderR(BinarySortTreePtr tree, void (*visit)(NodePtr)) 
{
    if (tree) preorder(tree->root, visit);
    return tree ? succeed : failed;
}

// 非递归中序遍历
Status BST_inorderI(BinarySortTreePtr tree, void (*visit)(NodePtr)) 
{
    if (!tree || !tree->root) return failed;

    Stack* stack = createStack(100);
    NodePtr current = tree->root;

    while (current || stack->top != -1) 
    {
        while (current) 
        {
            push(stack, current);
            current = current->left;
        }
        current = pop(stack);
        visit(current);
        current = current->right;
    }

    free(stack->data);
    free(stack);
    return succeed;
}

static void inorder(NodePtr node, void (*visit)(NodePtr)) 
{
    if (node) 
    {
        inorder(node->left, visit);
        visit(node);
        inorder(node->right, visit);
    }
}

Status BST_inorderR(BinarySortTreePtr tree, void (*visit)(NodePtr))
{
    if (tree) inorder(tree->root, visit);
    return tree ? succeed : failed;
}

// 非递归后序遍历
Status BST_postorderI(BinarySortTreePtr tree, void (*visit)(NodePtr)) 
{
    if (!tree || !tree->root) return failed;

    Stack* stack1 = createStack(100);
    Stack* stack2 = createStack(100);
    push(stack1, tree->root);

    while (stack1->top != -1) 
    {
        NodePtr node = pop(stack1);
        push(stack2, node);
        if (node->left) push(stack1, node->left);
        if (node->right) push(stack1, node->right);
    }

    while (stack2->top != -1) 
    {
        visit(pop(stack2));
    }

    free(stack1->data);
    free(stack1);
    free(stack2->data);
    free(stack2);
    return succeed;
}

static void postorder(NodePtr node, void (*visit)(NodePtr)) 
{
    if (node) 
    {
        postorder(node->left, visit);
        postorder(node->right, visit);
        visit(node);
    }
}

Status BST_postorderR(BinarySortTreePtr tree, void (*visit)(NodePtr))
{
    if (tree) postorder(tree->root, visit);
    return tree ? succeed : failed;
}

// 层序遍历
Status BST_levelOrder(BinarySortTreePtr tree, void (*visit)(NodePtr)) 
{
    if (!tree || !tree->root) return failed;

    NodePtr queue[100];
    int front = 0, rear = 0;
    queue[rear++] = tree->root;

    while (front < rear) 
    {
        NodePtr node = queue[front++];
        visit(node);
        if (node->left) queue[rear++] = node->left;
        if (node->right) queue[rear++] = node->right;
    }
    return succeed;
}