import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import sqlite3
import os

class SQLiteViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("SQLite 数据库查看器")
        self.root.geometry("800x600")
        
        # 当前数据库连接
        self.conn = None
        self.current_table = None
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开数据库", command=self.open_database)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 数据库信息框架
        info_frame = ttk.LabelFrame(main_frame, text="数据库信息")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.db_path_label = ttk.Label(info_frame, text="未选择数据库")
        self.db_path_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 表选择框架
        table_frame = ttk.LabelFrame(main_frame, text="表选择")
        table_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.table_combo = ttk.Combobox(table_frame, state="readonly")
        self.table_combo.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_selected)
        
        refresh_btn = ttk.Button(table_frame, text="刷新", command=self.refresh_tables)
        refresh_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 数据表格框架
        data_frame = ttk.LabelFrame(main_frame, text="表数据")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建带滚动条的表格
        self.create_table_with_scrollbar(data_frame)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_table_with_scrollbar(self, parent):
        # 创建滚动条
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL)
        
        # 创建表格
        self.tree = ttk.Treeview(parent, 
                                yscrollcommand=v_scrollbar.set, 
                                xscrollcommand=h_scrollbar.set)
        
        # 配置滚动条
        v_scrollbar.config(command=self.tree.yview)
        h_scrollbar.config(command=self.tree.xview)
        
        # 布局
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # 配置网格权重
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        # 绑定双击事件
        self.tree.bind('<Double-1>', self.on_cell_double_click)
    
    def open_database(self):
        file_path = filedialog.askopenfilename(
            title="选择SQLite数据库文件",
            filetypes=[("SQLite数据库", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 关闭现有连接
                if self.conn:
                    self.conn.close()
                
                # 连接新数据库
                self.conn = sqlite3.connect(file_path)
                self.db_path_label.config(text=f"数据库: {os.path.basename(file_path)}")
                self.status_var.set(f"已连接到数据库: {file_path}")
                
                # 加载表列表
                self.load_tables()
                
            except sqlite3.Error as e:
                messagebox.showerror("数据库错误", f"无法打开数据库:\n{str(e)}")
    
    def load_tables(self):
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.table_combo['values'] = tables
            
            if tables:
                self.table_combo.current(0)
                self.on_table_selected()
            else:
                self.clear_table()
                self.status_var.set("数据库中没有表")
                
        except sqlite3.Error as e:
            messagebox.showerror("数据库错误", f"无法加载表列表:\n{str(e)}")
    
    def refresh_tables(self):
        if self.conn:
            self.load_tables()
    
    def on_table_selected(self, event=None):
        if not self.conn:
            return
            
        table_name = self.table_combo.get()
        if table_name:
            self.current_table = table_name
            self.load_table_data(table_name)
    
    def load_table_data(self, table_name):
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # 清空现有表格
            self.clear_table()
            
            # 设置表格列
            self.tree['columns'] = column_names
            self.tree.heading('#0', text='ID')
            self.tree.column('#0', width=50, minwidth=50)
            
            for col in column_names:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, minwidth=50)
            
            # 获取表数据
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # 插入数据
            for i, row in enumerate(rows):
                self.tree.insert('', 'end', text=str(i+1), values=row)
            
            self.status_var.set(f"表 '{table_name}' 已加载，共 {len(rows)} 行")
            
        except sqlite3.Error as e:
            messagebox.showerror("数据库错误", f"无法加载表数据:\n{str(e)}")
    
    def clear_table(self):
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 清空列定义
        for col in self.tree['columns']:
            self.tree.heading(col, text='')
    
    def on_cell_double_click(self, event):
        if not self.conn or not self.current_table:
            return
            
        # 获取选中的项目
        item = self.tree.selection()[0] if self.tree.selection() else None
        if not item:
            return
            
        # 获取列ID
        column = self.tree.identify_column(event.x)
        if column == '#0':  # 行号列，不允许编辑
            return
            
        # 获取列索引
        col_index = int(column.replace('#', '')) - 1
        
        # 获取当前值
        current_value = self.tree.item(item, 'values')[col_index]
        
        # 获取列名
        column_name = self.tree['columns'][col_index]
        
        # 创建自定义编辑对话框
        self.create_edit_dialog(item, column_name, current_value)
    
    def create_edit_dialog(self, item, column_name, current_value):
        # 创建新窗口
        edit_dialog = tk.Toplevel(self.root)
        edit_dialog.title(f"编辑 '{column_name}'")
        edit_dialog.geometry("500x300")  # 设置更大的默认大小
        edit_dialog.transient(self.root)
        edit_dialog.grab_set()
        
        # 设置对话框图标和位置
        edit_dialog.resizable(True, True)
        
        # 创建框架
        main_frame = ttk.Frame(edit_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 列名标签
        label = ttk.Label(main_frame, text=f"列: {column_name}")
        label.pack(anchor=tk.W, pady=(0, 5))
        
        # 创建带滚动条的文本框
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_scrollbar = ttk.Scrollbar(text_frame)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_area = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=text_scrollbar.set)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.config(command=text_area.yview)
        
        # 插入当前值
        text_area.insert('1.0', str(current_value) if current_value is not None else "")
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 确定按钮
        ok_button = ttk.Button(button_frame, text="确定", 
                              command=lambda: self.save_edit(item, column_name, text_area.get('1.0', tk.END).strip(), edit_dialog))
        ok_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 取消按钮
        cancel_button = ttk.Button(button_frame, text="取消", 
                                  command=edit_dialog.destroy)
        cancel_button.pack(side=tk.RIGHT)
        
        # 绑定回车键和ESC键
        edit_dialog.bind('<Return>', lambda e: ok_button.invoke())
        edit_dialog.bind('<Escape>', lambda e: cancel_button.invoke())
        
        # 让文本框获得焦点
        text_area.focus_set()
    
    def save_edit(self, item, column_name, new_value, dialog):
        if not self.conn or not self.current_table:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # 获取主键列名
            cursor.execute(f"PRAGMA table_info({self.current_table})")
            columns = cursor.fetchall()
            primary_key_col = None
            
            for col in columns:
                if col[5] == 1:  # 主键标志
                    primary_key_col = col[1]
                    break
            
            if not primary_key_col:
                messagebox.showerror("更新错误", "无法确定主键列")
                return
            
            # 获取当前行的主键值
            item_values = self.tree.item(item, 'values')
            col_names = self.tree['columns']
            primary_key_value = None
            
            for i, col in enumerate(col_names):
                if col == primary_key_col:
                    primary_key_value = item_values[i]
                    break
            
            if primary_key_value is None:
                messagebox.showerror("更新错误", "无法找到主键值")
                return
            
            # 执行更新
            cursor.execute(
                f"UPDATE {self.current_table} SET {column_name} = ? WHERE {primary_key_col} = ?",
                (new_value, primary_key_value)
            )
            self.conn.commit()
            
            # 更新表格显示
            values = list(self.tree.item(item, 'values'))
            col_index = col_names.index(column_name)
            values[col_index] = new_value
            self.tree.item(item, values=values)
            
            self.status_var.set(f"单元格已更新")
            
            # 关闭对话框
            dialog.destroy()
            
        except sqlite3.Error as e:
            messagebox.showerror("数据库错误", f"更新失败:\n{str(e)}")
            self.conn.rollback()

def main():
    root = tk.Tk()
    app = SQLiteViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()