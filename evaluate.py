import time
from rag_engine import RAGSystem
from tabulate import tabulate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_evaluation():
    print("Loading RAG System...")
    rag = RAGSystem()
    rag.load_vector_store()

    judge_llm = rag.llm 
    
    judge_prompt = ChatPromptTemplate.from_template(
        "Compare the answer to the expected fact. Return ✅ for correct/safe refusal, ❌ for hallucination. \nFact: {expected}\nAnswer: {actual}"
    )
    grader_chain = judge_prompt | judge_llm | StrOutputParser()

    eval_set = [
        {
            "category": "Fact",
            "question": "What is the deadline for submitting a return request?",
            "expected": "10 days from date of delivery"
        },
        {
            "category": "Complex Logic",
            "question": "If I cancel a service order after 9 days, how much is the fee?",
            "expected": "9% deduction from the total amount"
        }
    ]

    results = []
    print("\n--- 📝 COMPARING BASIC VS ADVANCED PROMPTS ---\n")

    for item in eval_set:
        print(f"Testing: {item['question']}...")
        
        # 1. Test Basic
        res_basic = rag.query(item['question'], version="basic")
        ans_basic = res_basic.get('answer', "Error")
        time.sleep(2) # Safety pause

        # Judge Basic
        grade_basic = grader_chain.invoke({"expected": item['expected'], "actual": ans_basic}).strip()
        time.sleep(2) # Safety pause

        # 2. Test Advanced
        res_adv = rag.query(item['question'], version="advanced")
        ans_adv = res_adv.get('answer', "Error")
        time.sleep(2) # Safety pause

        # Judge Advanced
        grade_adv = grader_chain.invoke({"expected": item['expected'], "actual": ans_adv}).strip()
        time.sleep(2) # Safety pause

        results.append([
            item['category'],
            item['question'][:20] + "...",
            f"{ans_basic[:40]}... ({grade_basic})",
            f"{ans_adv[:40]}... ({grade_adv})"
        ])

    print(tabulate(results, headers=["Cat", "Question", "Basic", "Advanced"], tablefmt="grid"))

if __name__ == "__main__":
    run_evaluation()