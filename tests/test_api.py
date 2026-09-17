import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure root folder is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

class TestDocVisionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.db.database import get_db_connection
        conn = get_db_connection()
        conn.execute("DELETE FROM users WHERE email LIKE '%@example.com'")
        conn.commit()
        conn.close()

        cls.client = TestClient(app)
        
        # Create a dummy image for upload testing
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.dummy_img_path = os.path.join(cls.test_dir, "api_test_invoice.jpg")
        
        from PIL import Image
        img = Image.new("RGB", (300, 400), color="white")
        img.save(cls.dummy_img_path)

    @classmethod
    def tearDownClass(cls):
        # Cleanup dummy image
        if os.path.exists(cls.dummy_img_path):
            os.remove(cls.dummy_img_path)

    def test_health_endpoint(self):
        """Tests the system health API."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("simulator_mode", data)
        self.assertIn("engine_specs", data)

    def test_metrics_endpoint(self):
        """Tests the training curves and VRAM metrics API."""
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("training_run", data)
        self.assertIn("optimization_comparison", data)
        self.assertEqual(data["training_run"]["base_model"], "Qwen/Qwen2.5-VL-7B-Instruct")

        # Register user and get token
        reg = self.client.post("/api/auth/register", json={
            "name": "API Test User",
            "email": "apitest@example.com",
            "password": "Password123",
            "confirm_password": "Password123"
        })
        token = reg.json()["access_token"]

        with open(self.dummy_img_path, "rb") as f:
            files = {"file": ("api_test_invoice.jpg", f, "image/jpeg")}
            data = {"doc_type": "invoice"}
            
            response = self.client.post(
                "/api/extract",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {token}"}
            )
            
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["document_type"], "invoice")
        self.assertIn("fields", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("processed_image_url", json_data)

    def test_predict_endpoint(self):
        """Tests direct prediction from a server local file path."""
        payload = {
            "image_path": self.dummy_img_path,
            "doc_type": "invoice"
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["document_type"], "invoice")
        self.assertIn("fields", json_data)
        self.assertIn("confidence", json_data)

    def test_train_endpoint(self):
        """Tests triggering the LoRA fine-tuning training job."""
        data = {
            "epochs": 1,
            "lr": 0.0002,
            "batch_size": 2,
            "grad_accum": 1
        }
        response = self.client.post("/api/train", data=data)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertTrue(json_data["success"])
        self.assertIn("message", json_data)
        self.assertEqual(json_data["state"]["status"], "triggered")

if __name__ == "__main__":
    unittest.main()
