import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
from utils import get_llm_creator, \
                    get_best_domain_llm, \
                    get_average_scores_for_llm, \
                    get_metrics_llm_domain, \
                    SCORE_COLUMNS
                    


# set page title and icon
st.set_page_config(page_title="LLM Leaderboard", page_icon="data/elv_symbol_avatar_1200x1200.png",
                   layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



@st.cache_data
def load_data(file_path:Path,sheet_name:str):

    df_llm_leaderboard = pd.read_excel(file_path,sheet_name)
    return df_llm_leaderboard


if __name__ == "__main__":

    file_path = "data/Leaderboard.xlsx"
    sheet_name = "Evaluation_type_view"
    df_llm_leaderboard = load_data(file_path,sheet_name)
    df_llm_leaderboard["creator"] = df_llm_leaderboard["LLM"].apply(get_llm_creator)
    df_llm_leaderboard = df_llm_leaderboard.rename(columns={"dataset":"domain","Evaluation_type":"task"})
    llms = df_llm_leaderboard["LLM"].unique()
    domains = df_llm_leaderboard["domain"].unique()
    # load CSS file
    local_css('leaderboard.css')

    # Header 
    header_col1, header_col2, header_col3,header_col4 = st.columns([1,1,8,1],gap="small")

    # streamlit dashboard
    with header_col2:
        st.image("data/elv_symbol_rgb_c.png",width=100)
    with header_col3:
        # st.markdown("# LLM Leaderboard")
        st.markdown('<div class="centered-markdown"><h1>LLM Leaderboard</h1></div>', unsafe_allow_html=True)


    st.divider()
    # Section 1

    sec1_col1, sec1_col2 = st.columns([2,2])
    with sec1_col1:

        st.markdown(
            """
            <div class="custom-text">
                <p class="section2-text">With the increasing number of Large Language Models being made available (both open and closed source), finding a suitable model for a use case can be challenging and time consuming as it requires extensive experimentation.
                This LLM Leaderboard evaluates and ranks all LLMs available within the Enterprise (ESG approved for use within the Enterprise), and provides reproducible scores offering valuable insights to AI teams during their LLM selection process for their initiative.
                Unlike other leaderboards widely available on internet, we will use healthcare industry data for evaluation. This will reflect the true effectiveness of an LLM on our company specific use cases and data.</p>
                <p class="section2-note"><strong>Note:</strong> At present the evaluations are done only for single modality (text).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with sec1_col2:
        st.image("data/bg_image.jpg")

    # st.divider()

    # st.markdown(
    #     """
    #     <div class="content">
    #         <div class="custom-text">
    #             <p>With the increasing number of Large Language Models being made available (both open and closed source), finding a suitable model for a use case can be challenging and time consuming as it requires extensive experimentation.</p>
    #             <p>This LLM Leaderboard evaluates and ranks all LLMs available within the Enterprise (<strong>ESG approved</strong> for use within the Enterprise), and provides reproducible scores offering valuable insights to AI teams during their LLM selection process for their initiative.</p>
    #             <p>Unlike other leaderboards widely available on the internet, we will use <strong>healthcare industry data</strong> for evaluation. This will reflect the true effectiveness of an LLM on our company-specific use cases and data.</p>
    #             <p><strong>Note:</strong> At present the evaluations are done only for single modality (text).</p>
    #         </div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.divider()


    # section 2 - show all the llms in as buttons or blocks

    st.markdown(
        """
    ### LLMs available for use within the Enterprise (ESG approved) 
    """
    )

    # st.data_editor(df_llm_leaderboard.head())

    llms= df_llm_leaderboard['LLM'].unique()
    sec2_col1, sec2_col2, sec2_col3 = st.columns(3)

    for i in range(len(llms)):
        if(i%3==0):
            with sec2_col1:
                st.button(llms[i])
        if(i%3==1):
            with sec2_col2:
                st.button(llms[i])

        if(i%3==2):
            with sec2_col3:
                st.button(llms[i]) 


    st.divider()

    # section 3

    st.markdown(""" ### Most suitable domain for an LLM
    """)    
    df_highest_scoring_domain = get_best_domain_llm(df_llm_leaderboard)
    st.table(df_highest_scoring_domain)

    st.divider()

    # section 4:
    with st.container():
        st.markdown("""
        ### Select two models and a domain to compare the model performance side by side across various tasks in the domain 
        """)    
        
        sel_domain = st.selectbox(label="Select a Domain",options=domains)

        sec4_col1, sec4_col2 = st.columns(2)

        with sec4_col1:
            sel_llm1 = st.selectbox(label="Select an LLM to compare",key="llm_select_box_1",options=llms)
            with st.container():
                df_llm_average_score_per_task = get_average_scores_for_llm(df_llm_leaderboard,sel_llm1,sel_domain)
                st.table(df_llm_average_score_per_task)

        
        with sec4_col2:
            sel_llm2 = st.selectbox(label="Select an LLM to compare",key="llm_select_box_2",options=[x for x in llms if x != sel_llm1])
            with st.container():
                df_llm_average_score_per_task = get_average_scores_for_llm(df_llm_leaderboard,sel_llm2,sel_domain)
                st.table(df_llm_average_score_per_task)

    st.divider()

    # section 5:

    st.markdown("""
    ### Select models and a domain to compare the model performance across various tasks in the domain for each metric. \
                These benchmarks assess the models' performance across all tasks \
                in a domain, and across various metrics. Different \
                metrics inform about different aspects of the behavior of \
                the model.  
    """)

    sec5_col1, sec5_col2, sec5_col3 = st.columns(3)

    with sec5_col1:
        sec5_sel_llms = st.multiselect(label="Select LLM(s)",options=llms,default=llms[0],max_selections=5)
    with sec5_col2:
        sec5_sel_domain = st.selectbox(label="Select a domain",options=domains)
    with sec5_col3:
        sec5_sel_metrics = st.multiselect(label="Select Metric(s)",options=SCORE_COLUMNS,default=SCORE_COLUMNS[0])
    
    metric_dfs = get_metrics_llm_domain(df=df_llm_leaderboard,llms=sec5_sel_llms,domain=sec5_sel_domain,metrics=sec5_sel_metrics)

    st.markdown(f""" ### Domain : {sec5_sel_domain}""")
    for metric, df in metric_dfs.items():
        st.markdown(f""" #### Metric: {metric} """)
        st.table(df.unstack().reset_index())
        df = df.reset_index()
        # st.data_editor(df)
        st.markdown(f""" #### Average {metric} score per Task for {sec5_sel_domain}""")
        st.plotly_chart(px.bar(df, x="task",y=metric, color="LLM",barmode='group'))
    
    st.divider()

    st.markdown(""" ###  Additional Resources: 
                
    #### For questions please contact : shaheen.gauher@elevancehealth.com 
    #### How you can help Enterprise Gen AI journey - If you have evaluation datasets and would like to include your use case in the leaderboard, please contact us. 

    """)