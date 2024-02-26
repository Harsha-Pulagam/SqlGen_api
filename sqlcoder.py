from inference import Inference
inference = Inference()

class Sqlcoder:
    def __init__(self):
        pass

    def sql_generator(self, text:str, inference): 
        sql_statement = inference.generate_query(text)
        return sql_statement 
"""
if __name__ == "__main__":
    sql = Sqlcode()
    print(sql.sentiment_analysis("What was our revenue by product in the New York region last month?", classifier))
    """
    