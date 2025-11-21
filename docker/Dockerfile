ARG PYTORCH_IMAGE
FROM ${PYTORCH_IMAGE}-devel as builder

WORKDIR /tmp/tinify
COPY tinify.tar.gz .
RUN pip install pybind11
RUN tar xzf tinify.tar.gz && \
		python3 setup.py bdist_wheel

FROM ${PYTORCH_IMAGE}-runtime

LABEL maintainer="tinify@interdigital.com"

WORKDIR /tmp
COPY --from=builder /tmp/tinify/dist/tinify-*.whl .
RUN pip install tinify-*.whl && \
		python3 -c 'import tinify'

# Install jupyter?
ARG WITH_JUPYTER=0
RUN if [ "$WITH_JUPYTER" = "1" ]; then \
		pip3 install jupyter ipywidgets && \
		jupyter nbextension enable --py widgetsnbextension \
		; fi

WORKDIR /workspace
CMD ["bash"]
