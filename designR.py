__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from pyDOE2.doe_factorial import *
from pyDOE2.doe_composite import *
from pyDOE2.doe_box_behnken import *
from PIL import Image
import base64
from io import BytesIO
import xlsxwriter

image = Image.open("logo.png")


def compute_table(M, low, high):
    """
    Transforms orthogonal design to real values.

    Inputs:
    M, orthogonal design
    low, values of lowest levels
    high, values of highest levels

    Output:
    Matrix with real values.
    """
    M_s = (M-np.min(M, axis=0))/(np.max(M, axis=0)-np.min(M, axis=0))
    low = np.asarray(low)
    high = np.asarray(high)
    delta = high-low
    return np.add(np.multiply(M_s, delta), low)


def to_excel(DataFrame):
    """
    Creates excel file.

    Inputs:
    pd.DataFrame

    Outputs:
    Excel-file
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    DataFrame.to_excel(writer, sheet_name="Sheet1")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(DataFrame):
    """
    Download Excel-file.

    Inputs:
    pd.DataFrame

    Outputs:
    Download of file.
    """
    val = to_excel(DataFrame)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="design.xlsx">Download design protocol.</a>'


def factorial_design(n_factors):
    """
    Create two-level full factorial design.

    Inputs:
    n_factors, number of factors

    Outputs:
    design matrix
    """
    return ff2n(n_factors)


def fractorial_design(n_factors):
    """
    Create two-level fractional factorial design.

    Inputs:
    n_factors, number of factors

    Outputs:
    design matrix
    """
    generator = "A B C D E F G H I J K L M N O P Q R S T"
    gens = st.sidebar.text_input("""design generators
                                 (separated by spaces)""",
                                 value=generator[0:2*n_factors])

    n_generators = len((gens[:-1] if gens[-1] == " " else gens).split(" "))

    try:
        assert n_generators == n_factors
    except AssertionError:
        st.error("Number of generators does not match number of factors.")
        st.stop()
    return fracfact(gens)


def composite_design(n_factors):
    """
    Create central composite design.

    Inputs:
    n_factors, number of factors

    Outputs:
    design matrix
    """
    alpha = st.sidebar.selectbox("symmetry",
                                 options=["orthogonal", "rotatable"])

    face = st.sidebar.selectbox("geometry",
                                options=["circumscribed",
                                         "inscribed",
                                         "faced"])

    n_c1 = st.sidebar.number_input("""number of center points in
                                   the factorial block""", min_value=4)

    n_c2 = st.sidebar.number_input("""number of center points in
                                   the star block""", min_value=4)

    return ccdesign(n_factors, (n_c1, n_c2), alpha, face)


def box_behnken_design(n_factors):
    """
    Create Box-Behnken design.

    Inputs:
    n_factors, number of factors

    Outputs:
    design matrix
    """
    n_c1 = st.sidebar.number_input("""number of center points""",
                                   min_value=4)

    return bbdesign(n_factors, n_c1)


def main():
    st.sidebar.text("© Georgi Tancev")

    st.image(image, caption="""design-R is an application
    for design of experiments.""", use_column_width=True)

    st.subheader("Read instructions.")
    st.write("""
             1. Set number of factors.
             2. Set labels, minimum, and maximum values of factors.
             3. Choose design class and customize it.
             4. Inspect design for correctness.
             5. Download design protocol.
             """)

    st.subheader("Set number of factors.")

    n_factors = st.slider("number of factors",
                          min_value=1,
                          max_value=10,
                          value=4, step=1)

    st.subheader("Set factor names and levels.")

    col1, col2, col3 = st.beta_columns(3)

    labels = []
    low = []
    high = []

    for i in range(1, n_factors+1):

        with col1:
            labels.append(col1.text_input("factor name ("+str(i)+")",
                                          value=str(i)))

        with col2:
            low.append(col2.number_input("minimum level ("+str(i)+")",
                                         value=-1))

        with col3:
            high.append(col3.number_input("maximum level ("+str(i)+")",
                                          value=1))

    st.sidebar.subheader("Customize experimental design.")

    rescaled = st.sidebar.checkbox("rescale levels", value=True)

    design = st.sidebar.selectbox("design class",
                                  options=["full factorial",
                                           "fractional factorial",
                                           "central composite",
                                           "Box-Behnken"],
                                  index=0)

    if design == "full factorial":
        M = factorial_design(n_factors)
    elif design == "fractional factorial":
        try:
            M = fractorial_design(n_factors)
        except IndexError:
            st.error("""One or more higher order design generators
                     do not match base designs.""")
            st.stop()
    elif design == "central composite":
        M = composite_design(n_factors)
    else:
        M = box_behnken_design(n_factors)

    st.subheader("Inspect experimental design.")

    for k in range(0, n_factors):
        M = M[np.argsort(M[:, k], axis=0, kind="stable"), :]
    if rescaled:
        data = compute_table(M, low, high)
    else:
        data = M
    table = pd.DataFrame(data=data,
                         index=np.arange(1, M.shape[0]+1),
                         columns=labels)
    table.index.name = "Experiment"
    st.table(table)

    st.markdown(get_table_download_link(table), unsafe_allow_html=True)

    return


main()
