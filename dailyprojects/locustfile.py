from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 2)  # seconds each user waits between tasks

    @task
    def get_root(self):
        self.client.get("/")
