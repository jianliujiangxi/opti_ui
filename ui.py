import functools
from pathlib import Path

import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.shared import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import psycopg2
import numpy as np
import plotly.express as px
import time

chart = functools.partial(st.plotly_chart, use_container_width=True)
COMMON_ARGS = {
	"color": "symbol",
	"color_discrete_sequence": px.colors.sequential.Greens,
	"hover_data": [
		"account_name",
		"percent_of_account",
		"quantity",
		"total_gain_loss_dollar",
		"total_gain_loss_percent",
	],
}


@st.experimental_memo
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Take Raw Fidelity Dataframe and return usable dataframe.
	- snake_case headers
	- Include 401k by filling na type
	- Drop Cash accounts and misc text
	- Clean $ and % signs from values and convert to floats
	Args:
		df (pd.DataFrame): Raw fidelity csv data
	Returns:
		pd.DataFrame: cleaned dataframe with features above
	"""
	df = df.copy()
	df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")

	df.type = df.type.fillna("unknown")
	df = df.dropna()

	price_index = df.columns.get_loc("last_price")
	cost_basis_index = df.columns.get_loc("cost_basis_per_share")
	df[df.columns[price_index : cost_basis_index + 1]] = df[
		df.columns[price_index : cost_basis_index + 1]
	].transform(lambda s: s.str.replace("$", "").str.replace("%", "").astype(float))

	quantity_index = df.columns.get_loc("quantity")
	most_relevant_columns = df.columns[quantity_index : cost_basis_index + 1]
	first_columns = df.columns[0:quantity_index]
	last_columns = df.columns[cost_basis_index + 1 :]
	df = df[[*most_relevant_columns, *first_columns, *last_columns]]
	return df


@st.experimental_memo
def filter_data(
	df, account_selections, symbol_selections
):
	"""
	Returns Dataframe with only accounts and symbols selected
	Args:
		df (pd.DataFrame): clean fidelity csv data, including account_name and symbol columns
		account_selections (list[str]): list of account names to include
		symbol_selections (list[str]): list of symbols to include
	Returns:
		pd.DataFrame: data only for the given accounts and symbols
	"""
	df = df.copy()
	df = df[
		df.account_name.isin(account_selections) & df.symbol.isin(symbol_selections)
	]

	return df

@st.experimental_memo
def preprocessing(df, _my_bar):
	params = {}
	for i in range(100):
		_my_bar.progress(i+1)
	return params   

def main() -> None:
	st.header("医疗资源配置优化系统")

	with st.expander("帮助文档"):
	  st.write(Path("README.md").read_text())

	st.subheader("请连接数据库")
	col_db, col_host, col_user, col_pwd, col_port = st.columns(5)
	with col_db:
		db = st.text_input('数据仓库', 'warehouse')
	with col_host:	
		host = st.text_input('主机名', '172.16.4.70', type='password')
	with col_user:	
		user = st.text_input('用户名', 'jian_liu')
	with col_pwd:	
		password = st.text_input('密码', 'jian_liu_mjFS21.com', type='password')
	with col_port:	
		port = st.text_input('端口', '5432', type='password')

	col_table, col_hospital, col_record_year = st.columns([0.5, 0.3, 0.2])
	with col_table:
		table_list = ['biuse.jianhao_ror_datasource_dept',
				'biuse.jianhao_ror_datasource_drggroup']
		table = st.multiselect("数据表名称", options=table_list, default=table_list)
	with col_hospital:	
		hospital = st.text_input('医院名称', '湘潭市第一人民医院')
	with col_record_year:
		record_year = st.text_input('报告年份', '2021')

	connect = st.button('连接数据库')	
	if connect:
		with st.spinner('等待数据库连接......'):	
			try:
				cur = psycopg2.connect(
									   database=db,
									   user=user,
									   password=password,
									   host=host,
									   port=port
										)
			except:	
				raise ValueError('数据库连接失败')
			tables = {}
			for t in table:
				sql_code = "select * FROM %s WHERE hospital = '%s' and record_year = '%s'" % (t, hospital, record_year)
				try:
					tables[t.split('.')[1]] = pd.read_sql(sql_code, con=cur)
				except:	
					raise ValueError('读取表 '+ t + ' 失败')
		st.success('数据库连接成功')				

	uploaded_data = open("example.csv", "r")
	df = pd.read_csv(uploaded_data)
	with st.expander("Raw Dataframe"):
		st.write(df)

	df = clean_data(df)
	with st.expander("Cleaned Data"):
		st.write(df)

	st.sidebar.header('可交互界面') 
	st.sidebar.subheader("过滤不输入模型的科室和病组")

	accounts = list(df.account_name.unique())
	account_selections = st.sidebar.multiselect(
		"过滤特殊科室", options=accounts, default=None
	)

	symbols = list(df.loc[df.account_name.isin(account_selections), "symbol"].unique())
	symbol_selections = st.sidebar.multiselect(
		"过滤特殊病组", options=symbols, default=None
	)

	df = filter_data(df, account_selections, symbol_selections)
	# 过滤完数据之后，接着各种处理数据，综合成一步得到符合模型输入的初步params
	st.subheader("数据预处理")
	my_bar = st.progress(0)
	params = preprocessing(df, my_bar)
	my_bar.progress(100)
	st.success('数据已处理完毕')

	st.sidebar.subheader('选择模型的优化目标')
	DRG = st.sidebar.checkbox('最大化DRG总权重', 1)
	CMI = st.sidebar.checkbox('最大化CMI值', 0)

	st.sidebar.subheader('选择模型种类')
	if (DRG and not CMI) or (not DRG and CMI): 
		single_model_type = st.sidebar.radio('模型类别',
								 options=['单目标规划', '单目标的目标规划'],
								 index=0)
		if single_model_type == '单目标的目标规划' and DRG:
			st.sidebar.number_input('期望DRG总权重值')
		if single_model_type == '单目标的目标规划' and CMI:
			st.sidebar.number_input('期望CMI值')
	elif DRG and CMI:	
		multiple_model_type = st.sidebar.radio('模型类别',
								 options=['多目标规划', '多目标的目标规划'],
								 index=0)
		if multiple_model_type == '多目标的目标规划':
			st.sidebar.number_input('期望DRG总权重值', value=70000)
			st.sidebar.number_input('期望CMI值', value=1.4)

	st.sidebar.subheader('选择模型的决策变量')
	drg = st.sidebar.checkbox('DRG病组', 1)
	bed = st.sidebar.checkbox('科室病床数', 1)
	los = st.sidebar.checkbox('平均住院日', 0)

	st.sidebar.subheader('选择模型变量box-bound的约束')
	lb_ub_ratio = [round(i*0.05, 2) for i in range(41)]
	los_lb_ub = [round(i*0.1, 2) for i in range(301)]
	if drg:
		basic_drg_lbub_cons = st.sidebar.checkbox('病组的病例数约束', 1)
		if basic_drg_lbub_cons:
			st.sidebar.select_slider('基础病组上下限', options=lb_ub_ratio, value=(0.0, 1.1))
			st.sidebar.select_slider('奇异病组上下限', options=lb_ub_ratio, value=(0.0, 1.15))
			st.sidebar.select_slider('普通病组上下限', options=lb_ub_ratio, value=(0.8, 1.2))
			st.sidebar.select_slider('非主要病组上下限', options=lb_ub_ratio, value=(0.0, 1.1))
	if bed:	
		bed_lbub_cons = st.sidebar.checkbox('科室床位数约束', 1)
		if bed_lbub_cons:
			st.sidebar.select_slider('床位数上下限', options=lb_ub_ratio, value=(0.9, 1.1))
	if los:	
		los_lb_ub_cons = st.sidebar.checkbox('病组平均住院日约束', 1)
		if los_lb_ub_cons:
			st.sidebar.select_slider('平均住院日上下限', options=los_lb_ub, value=(0.8, 1.0))

	st.sidebar.subheader('选择模型业务约束')
	beds_cons = st.sidebar.checkbox('医院总床位数约束', value=1)
	dept_bed_util_cons = st.sidebar.checkbox('科室病床使用率约束', value=1)
	hosp_bed_util_cons = st.sidebar.checkbox('医院病床使用率约束', value=1)
	case_cons = st.sidebar.checkbox('总病例数约束', value=1)
	drg_weight_cons= st.sidebar.checkbox('DRG总权重约束', value=1)
	cmi_cons = st.sidebar.checkbox('CMI值的上下限约束', value=1)
	los_cons = st.sidebar.checkbox('医院最大平均住院日', value=1)

	st.sidebar.subheader('模型参数调控')

	if beds_cons:
		st.sidebar.slider('医院总床位数', 1000, 3000)
	if dept_bed_util_cons:	
		st.sidebar.select_slider('科室病床使用率上下限', options=lb_ub_ratio, value=(0.8, 0.95))
	if hosp_bed_util_cons:	
		st.sidebar.select_slider('医院病床使用率上下限', options=lb_ub_ratio, value=(0.8, 0.95))
	if case_cons:	
		st.sidebar.select_slider('总病例数上下限', options=lb_ub_ratio, value=(0.0, 1.0))
	if drg_weight_cons:	
		st.sidebar.select_slider('DRG总权重上下限', options=lb_ub_ratio, value=(0.0, 1.0))
	if cmi_cons:
		st.sidebar.select_slider('CMI上下限', options=lb_ub_ratio, value=(1.0, 1.6))
	if los_cons:	
		st.sidebar.select_slider('医院平均住院日上下限', options=los_lb_ub, value=(8.0, 13.0))


	st.subheader("Selected Account and Ticker Data")
	cellsytle_jscode = JsCode(
		"""
	function(params) {
		if (params.value > 0) {
			return {
				'color': 'white',
				'backgroundColor': 'forestgreen'
			}
		} else if (params.value < 0) {
			return {
				'color': 'white',
				'backgroundColor': 'crimson'
			}
		} else {
			return {
				'color': 'white',
				'backgroundColor': 'slategray'
			}
		}
	};
	"""
	)

	gb = GridOptionsBuilder.from_dataframe(df)
	gb.configure_columns(
		(
			"last_price_change",
			"total_gain_loss_dollar",
			"total_gain_loss_percent",
			"today's_gain_loss_dollar",
			"today's_gain_loss_percent",
		),
		cellStyle=cellsytle_jscode,
	)
	gb.configure_pagination()
	gb.configure_columns(("account_name", "symbol"), pinned=True)
	gridOptions = gb.build()

	AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)

	def draw_bar(y_val: str) -> None:
		fig = px.bar(df, y=y_val, x="symbol", **COMMON_ARGS)
		fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
		chart(fig)

	account_plural = "s" if len(account_selections) > 1 else ""
	st.subheader(f"Value of Account{account_plural}")
	totals = df.groupby("account_name", as_index=False).sum()
	if len(account_selections) > 1:
		st.metric(
			"Total of All Accounts",
			f"${totals.current_value.sum():.2f}",
			f"${totals.total_gain_loss_dollar.sum():.2f}",
		)
	for column, row in zip(st.columns(len(totals)), totals.itertuples()):
		column.metric(
			row.account_name,
			f"${row.current_value:.2f}",
			f"${row.total_gain_loss_dollar:.2f}",
		)

	fig = px.bar(
		totals,
		y="account_name",
		x="current_value",
		color="account_name",
		color_discrete_sequence=px.colors.sequential.Greens,
	)
	fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
	chart(fig)

	st.subheader("Value of each Symbol")
	draw_bar("current_value")

	st.subheader("Value of each Symbol per Account")
	fig = px.sunburst(
		df, path=["account_name", "symbol"], values="current_value", **COMMON_ARGS
	)
	fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
	chart(fig)

	st.subheader("Value of each Symbol")
	fig = px.pie(df, values="current_value", names="symbol", **COMMON_ARGS)
	fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
	chart(fig)

	st.subheader("Total Value gained each Symbol")
	draw_bar("total_gain_loss_dollar")
	st.subheader("Total Percent Value gained each Symbol")
	draw_bar("total_gain_loss_percent")


if __name__ == "__main__":
	st.set_page_config(
		"医院运营优化管理系统",
		"📊",
		initial_sidebar_state="expanded",
		layout="wide",
	)
	main()