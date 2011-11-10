'''
Generic base classes for data sources. The concept of being able to produce
infinitely many samples from an underlying distribution (or some other
production rule) is manifested by the MDP-compatible class DataSource.

The classes are abstract enough to represent any kind of data -- from images
to time series. The only limitation is that each sample should have the 
same number of elements (dimension).

@copyright: 
    2009-2011, Samuel John
@author: 
    Samuel John.
@contact: 
    www.SamuelJohn.de
@license: 
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

from datasource import (DataSource, DataSourceException, NoMoreSamplesException, 
                        FlowDataSource, SeededDataSource, ProbabilityDataSource,
                        DensityDataSource, CompositeDataSource, CascadedDataSource,
                        NoDuplicatesCascadedDataSource, TrainingSetDataSource,
                        TestSetDataSource, ImageDataSource, ImageFilesDataSource,
                        ImageFilesObjViewDataSource, UniformDataSource,
                        NoisyFigure3dDataSource )

__version__   = '0.3'
#__revision__  = utils.get_git_revision()
__authors__   = 'Samuel John'
__copyright__ = '2009-2011, Samuel John'
__license__   = 'APACHE 2.0, http://www.apache.org/licenses/LICENSE-2.0'
__contact__   = 'mail@SamuelJohn.de'
#__homepage__  = add github page here

# todo: do not import everything from datasource but only needed stuff
