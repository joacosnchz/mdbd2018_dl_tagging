SETLOCAL ENABLEDELAYEDEXPANSION
SET BATCH_SIZE=0


rem batch_size, epochs per training, using, categories, index

FOR /L %%A IN (1,1,10) DO (
	SET /A BATCH_SIZE=!BATCH_SIZE!+10
	python .\tags_net.py !BATCH_SIZE! 100 1 15 %%A
)