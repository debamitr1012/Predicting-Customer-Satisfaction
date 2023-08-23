# Predicting-Customer-Satisfaction

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
pip install -r requirements.txt
```

```bash
pip install zenml["server"]
zenml up
```

```bash
zenml integration install mlflow -y
```

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

```bash
python run_pipeline.py
```

```bash
python run_deployment.py
```

```bash
streamlit run streamlit_app.py
```
