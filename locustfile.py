from locust import HttpUser, task, between


class EdfApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        self.client.post("/predict", json={"date": "2024-12-15"})

    @task(2)
    def health(self):
        self.client.get("/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics")