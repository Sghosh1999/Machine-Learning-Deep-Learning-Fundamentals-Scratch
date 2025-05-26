from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List


class Book(BaseModel):
    name: str = Field(..., title="The name of the book", max_length=100)
    author: str = Field(..., title="The author of the book", max_length=100)
    year: Optional[int] = Field(None, title="The year the book was published")

app = FastAPI()

book_db = {}

@app.post("/books/", status_code=201)
async def create_book(book: Book):
    book_id = len(book_db) + 1
    book_db[book_id] = book
    return {"id": book_id, "book": book}

@app.get("/books/{book_id}", status_code=200)
async def get_book(book_id: int):
    if book_id in book_db:
        return {"id": book_id, "book": book_db[book_id]}
    raise HTTPException(status_code=404, detail="Book not found")

@app.put("/books/{book_id}", status_code=200)
async def update_book(book_id: int, book: Book):
    if book_id in book_db:
        book_db[book_id] = book
        return {"id": book_id, "book": book}
    raise HTTPException(status_code=404, detail="Book not found")

@app.get("/books/" ,status_code=200)
async def get_all_books():
    return {"books": [{"id": id, "book": book} for id, book in book_db.items()]}

@app.delete("/books/{book_id}", status_code=200)
async def delete_book(book_id: int):
    if book_id in book_db:
        deleted_book = book_db.pop(book_id)
        return {"message": "Book deleted successfully", "book": deleted_book}
    raise HTTPException(status_code=404, detail="Book not found")

