import json 
import unittest 

from app import app 

class IrisAPITest(unittest.TestCase): 
    def setUp(self):
        self.client = app.test_client() 
    
    def test_home(self):
        response = self.client.get("/")  
        self.assertEqual(response.status_code, 200)

    def test_predict_valid(self):
        payload = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        response = self.client.post("/predict", 
                                    data=json.dumps(payload),
                                    content_type="application/json") 
        self.assertEqual(response.status_code, 200) 
        result = response.get_json()
        self.assertIn("prediction", result) 

if __name__ == "__main__":
    unittest.main()