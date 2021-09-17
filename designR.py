__author__ = "Georgi Tancev"
__copyright__ = "© Georgi Tancev"

from itertools import combinations
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from pyDOE2.doe_factorial import *
from pyDOE2.doe_composite import *
from pyDOE2.doe_box_behnken import *
from pyDOE2.doe_plackett_burman import *
from PIL import Image
import base64
from io import BytesIO
import xlsxwriter


image = Image.open("logo.png")
design_image = Image.open("fullfact.png")


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
    alphabet = "A B C D E F G H I J K L M N O P Q R S T"
    length = int(np.ceil((n_factors+1)/2))
    letters = alphabet[:2*length:2]

    words = []
    for k in reversed(range(1, length+1)):
        for word in combinations(letters, k):
            words.append("".join(word[:]))

    generator = alphabet[:2*length]+" ".join(words[:(n_factors-length)])

    gens = st.sidebar.text_input("""design generators
                                 (separated by spaces)""",
                                 value=generator)

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
                                 options=["orthogonal", 
                                          "rotatable"])

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


def plackett_burman_design(n_factors):
    """
    Create Plackett-Burman design.

    Inputs:
    n_factors, number of factors

    Outputs:
    design matrix
    """
    return pbdesign(n_factors)


def main():

    st.set_page_config(page_title="design-R",
                       page_icon="🎨",
                       layout="centered",
                       initial_sidebar_state="auto")

    # st.sidebar.markdown(f'<a href="https://github.com/gtancev">© Georgi Tancev</a>',
                        # unsafe_allow_html=True)

    st.sidebar.subheader("Optional: Rescaling and sorting.")

    rescaled = st.sidebar.checkbox("Rescale levels.", value=False)
    sorted = st.sidebar.checkbox("Sort experiments.", value=True)

    st.sidebar.subheader("Choose design class.")

    design = st.sidebar.selectbox("design class",
                                  options=["full factorial",
                                           "fractional factorial",
                                           "Plackett-Burman",
                                           "central composite",
                                           "Box-Behnken"],
                                  index=2)

    st.image(image, use_column_width=True,
             caption="""design-R is a web application for
                the design of (laboratory) experiments.""")

    with st.expander(label="Read instructions."):

        st.write("""
                1. Set number of factors.
                2. Choose design class (and customize it).
                3. Optional: Set labels, minimum, and maximum values of factors.
                4. Inspect design for correctness.
                5. Download design protocol.
                """)

    st.subheader("Set number of factors.")

    n_factors = st.slider("number of factors",
                          min_value=2,
                          max_value=10,
                          value=7, step=1)

    st.subheader("Optional: Set labels and levels.")

    if not rescaled:
        st.info("""
                If rescale option is not selected, min./max. levels
                will not apply.
                """)

    col1, col2, col3 = st.columns(3)

    labels = np.empty(n_factors, dtype=np.object_)
    low = np.empty(n_factors)
    high = np.empty(n_factors)

    for i in range(1, n_factors+1):

        with col1:
            labels[i-1] = col1.text_input("label of factor "+str(i),
                                          value="x"+str(i))
        with col2:
            low[i-1] = col2.number_input("min. level of factor "+str(i),
                                         value=-1)
        with col3:
            high[i-1] = col3.number_input("max. level of factor "+str(i),
                                          value=1)

    if design == "full factorial":
        M = factorial_design(n_factors)
    elif design == "fractional factorial":
        try:
            M = fractorial_design(n_factors)
        except IndexError:
            st.error("""One or more higher order design generators
                     do not match base designs.""")
            st.stop()
    elif design == "Plackett-Burman":
        M = plackett_burman_design(n_factors)
    elif design == "central composite":
        M = composite_design(n_factors)
    else:
        try:
            M = box_behnken_design(n_factors)
        except AssertionError:
            st.error("""Number of factors must be at least three.""")
            st.stop()

    if rescaled:
        M = compute_table(M, low, high)
    if sorted:
        for k in range(0, n_factors):
            M = M[np.argsort(M[:, k], axis=0, kind="stable"), :]
    table = pd.DataFrame(data=M,
                         index=np.arange(1, M.shape[0]+1),
                         columns=labels)
    table.index.name = "Experiment"

    st.subheader("Inspect experimental design.")
    st.write("The number of experiments to be performed is",
             table.shape[0], ".")

    if sorted:
        st.info("""
                It is good practice to perform
                experiments in a random order.
                """)

    st.dataframe(table)

    st.download_button("Download design.",
                       to_excel(table),
                       "design.xlsx")

    with st.sidebar.expander(label="Read more about design of experiments."):
        st.markdown("""
                    The theory of experimental design proposes optimal
                    configurations of $k$ factors (i.e.,
                    independent variables $x_1$,
                    $x_2$, $...$, $x_k$) in order to asses their effect on
                    a dependent variable $y$ while minimizing
                    the number of experiments
                    that have to be performed. Nevertheless,
                    limiting the amount
                    of available data reduces the number of
                    parameters that can be included in a model,
                    thereby constraining model complexity.
                    """)
        st.image(design_image, use_column_width=True,
                 caption="Full factorial design for three factors.")

    return


main()
