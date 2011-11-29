'''
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

Created on Dec 29, 2009


@todo: Add test cases.
@todo: Add tester for subclasses of DataSource.
@todo: Make bimdp compatible ? 
'''
from __future__ import absolute_import, division, print_function
import mdp
import scipy as S
from scipy import where, zeros
import logging
import os



class DataSourceException(mdp.NodeException):
    '''Base Class for all DataSource Exceptions.'''


class NoMoreSamplesException(DataSourceException):
    """Raised when a data source cannot generate more samples."""


class NoMoreLabelsException(DataSourceException):
    """Raised when a data source cannot generate more labels.
    First draw further samples"""




class DataSource(mdp.Node):
    '''
    A class that represents a data source with samples of fixed
    dimensionality.

    Foremost, this class defines the methods sample() and samples(n).

    Additionally a data source D can be used as a mdp.Node. So, if you call
    the object as D() or call D.samples(n), the result is directly
    compatible to mdp's data format.
    So, the returned samples() are compatible to the convention of multiple
    "observations" the Modular Data Processing Toolkit defines.

    A data source can have any (fixed) dimensionality but each dimension
    should be scaled between 0.0 (incl.) and 1.0 (incl.) if you want to be
    able to use it as a density and combine different data-sources.
    However, you can implement your own logic and arbitrary ranges.

    sample() always has to return, so a datasource has to emit a sample or
    raise a NoMoreSamplesException. If the subclass's _sample() or _samples()
    returns None, NoMoreSamplesException is automatically raised.

    Therefore DataSource cannot be used to represent
    time series directly. If you do not want to have the time in discreet
    steps, you could provide such further information via get_labels().


    Methods to override:
    _sample()                  # Should increase _number_samples_until_now by 1

    Optionally:
    _samples()
    _get_labels()
    _reset()
    _get_supported_dtypes()
    _allows_duplicate_labels()  # default True. Overwrite with False if
                                # the datasource ensures that not two duplicate
                                # labels can be generated via sample
    _number_of_samples_max      # this is a property not a function. 
                                # Can be an int or scipy.Infinity and defines
                                # how many samples can be drawn from this DS.
                                # Defaults to infinity.

    Use, also:
    self._safemode              # To switch extra safety checks on/off (True/False)

    @see the class DemoDataSource

    @note: For random data, the SeededDataSource may be worth to inherit from
           and then use self.random instead of scipy.random.
    '''

    def __init__(self, output_dim=2, safemode=True, name="",
                 number_of_samples_execute=None,
                 number_of_samples_max=None,
                 loglevel=logging.INFO, **kws):
        '''
        @param output_dim:
            Declare the dimensionality of the samples. Default=2.
        @param safemode:
            Perform extra checks here and there. Default=True
        @param number_of_samples_execute:
            How many samples to draw when calling this DataSource as an
            mdp.Node. 
        @param name:
            An optional speaking name for this DS. Makes sense if you have
            the same type of data source, once for training and once for
            testing.
        @param number_of_samples_max: 
            The maximum number of samples this data source allows to be drawn.
            Normally you should leave this to default (None) to let the 
            data source decide.  
        '''
        super(DataSource,self).__init__(output_dim=output_dim, **kws)
        self.name                         = name
        self.add_logger(loglevel)
        self._safemode                    = safemode
        self._number_of_samples_until_now = 0
        self._last_label_nr               = 0
        
        if number_of_samples_max is not None:
            self._number_of_samples_max   = number_of_samples_max
        else:
            self._number_of_samples_max   = S.Infinity

        if number_of_samples_execute is None:
            if self.number_of_samples_max < S.Infinity:
                self.number_of_samples_execute = self.number_of_samples_max
            else:
                self.number_of_samples_execute = 1
        else:
            self.number_of_samples_execute     = number_of_samples_execute
            


    @property
    def ndim(self):
        '''@deprecated: Use sefl.output_dim (which is provided by mdp.Node)'''
        return self._output_dim


    @property
    def number_of_samples_until_now(self):
        '''How may samples have been drawn from this data source until now.'''
        return self._number_of_samples_until_now


    @property
    def number_of_samples_max(self):
        '''If the datasource is not infinite, this is an integer.
        Use number_of_samples_still_available() if you want to know if it
        is safe to call samples(n) with some n.'''
        return self._number_of_samples_max


    @property
    def number_of_samples_still_available(self):
        '''Defaults to number_of_samples_max - number_of_samples_until_now.
        But subclass may re-implement, if the number of available samples
        is not yet clear, then number_of_samples_max() must not necessarily
        be reached.
        Subclass may overwrite this one. It is used to find out how many
        sample can be drawn at each single samples() call.'''
        return self.number_of_samples_max - self.number_of_samples_until_now


    def _sample(self, **kws):
        raise NotImplementedError()


    def _samples(self, n=1, **kws):
        '''Draw n samples at once. May be more efficient in some cases.'''
        t = []
        for i in xrange(n):
            t.append( self._sample(**kws) )
        return t


    def _get_labels(self, n, start):
        '''Subclass may want to override and provide label information
        beginning from sample number start up to start+n (excl. the last one).
        It should return a list containing an element for each label.
        It's not (yet) defined, what a label should look like, but I think
        the best is fixed-length list of numbers representing the information
        that is needed to fully describe what has been generated. So to speak
        the latent variables.'''
        raise NotImplementedError()


    def _reset(self):
        '''Subclass may override this.'''
        pass


    def sample(self, **kws):
        '''Request this datasource to generate one sample. The output
        should obey the mdp convention, i.e. S.atleast_2d.
        Example: [[ 1,2,3,4 ]] is a sample with ouput_dim=4.

        Subclass should implement _sample and not overwrite this!'''
        self.log.debug('Requested 1 sample.')
        if self.number_of_samples_still_available < 1:
            raise NoMoreSamplesException('This data source is exhausted. It has already produced %i samples.' % self._number_of_samples_until_now)
        # old check:
        #if self._number_of_samples_until_now+1 > self.number_of_samples_max:
        #    raise NoMoreSamplesException('This data source is exhausted. It has already produced %i samples.' % self._number_of_samples_until_now)
        s = S.array(self._samples(n=1, **kws), dtype=self.dtype)
        if self._safemode:
            if len(s[0]) != self.output_dim:
                raise Exception('Dimension mismatch in DataSource output %i!=%i' % (len(s), self.output_dim))
        self._number_of_samples_until_now += 1
        return s


    def samples(self, n=1, **kws):
        '''Request n samples. Note that the subclass should overwrite
        _samples if it thinks that multiple can be generated more efficiently
        than calling _sample multiple times. The latter is done automatically.'''
        self.log.debug('Requested %i samples.',n)
        if n is S.Infinity:
            raise DataSourceException('You cannot get infinitely many samples at once.')
        if self.number_of_samples_still_available < n:
            #if self._number_of_samples_until_now+n > self.number_of_samples_max:
            raise NoMoreSamplesException('This data source is exhausted. It has already produced %i samples. Cannot draw n=%i additional samples.' % (self._number_of_samples_until_now, n))
        ss = S.array( self._samples(n, **kws), dtype=self.dtype)
        self._number_of_samples_until_now += n
        return ss


    def next(self):
        try:
            return self.sample()
        except NoMoreSamplesException,e:
            raise StopIteration(str(e))


    def all_remaining_samples(self):
        'Get all remaining samples or one samples if the DS is infinit.'
        return self.samples(n=self.number_of_samples_still_available)
        

    def all_samples(self, reset=True):
        '''Get all samples after self.reset if reset=True if the DS is finite.
        Or one samples if the DS is infinite.'''
        if reset: self.reset(verbose=False)
        return self.samples(n=self.number_of_samples_still_available)


    def _execute(self, x, n=None, **kws):
        '''
        MDP compatible call.

        @param x: 
            ignored.
        @param n:
            How many samples to draw and return.'''
        # we ignore x
        if n is None:
            n = self.number_of_samples_execute
        return self.samples(n, **kws)


    def get_labels(self,n=None, start=None, update_last_label_nr=True):
        '''
        Get the labels for the samples drawn so far.

        So first draw some samples. Then get the labels so far.
        @param  start:
            From which sample number to start. Because of DataSource being
            stateful, get_labels also remembers the position of the last
            labels you requested.
        @param end:
            Optionally. If given

        @return:
            A list of a list of entries, describing the generated sample(s).
            Each datasource can defines what elements are in each inner list.
            All lists should (but must not) have the same length. It's also not
            forbidden to contain dicts and other stuff in the inner list, but
            that is not recommended. See the doc of the subclass that
            actually implements (or not) the _get_labels() method.

        Subclass should implement _get_labels() and not overwrite this one!'''
        if start is None:
            start = self._last_label_nr
        if n is None:
            n = self.number_of_samples_until_now - start
        if start + n > self.number_of_samples_until_now:
            raise NoMoreLabelsException('More labels requested than samples have been drawn so far.')
        if update_last_label_nr:
            self._last_label_nr += n
        self.log.debug('get %i labels, beginning from start=%i', n, start)
        return self._get_labels(n=n, start=start)


    def reset(self,verbose=True):
        '''Resetting this datasource, so the first samples it returned again on next sample().'''
        if verbose: self.log.info('resetting.')
        self._number_of_samples_until_now = 0
        self._last_label_nr = 0
        self._reset() # give subclass a change to react


    def is_trainable(self):
        return False


    def is_invertible(self):
        return False


    @property
    def allows_duplicate_labels(self):
        '''Whether this data source allows duplicate labels or not.
        A subclass may overwrite this. When a datasource overwrites
        _allows_duplicate_labels to return False, then it has to ensure
        that (even if sample() uses a random gen.) not two of the same
        label parameters are returned.
        @note: It is *not* forbidden to produce the same sample even for
        different labels. Whether this is good design, is another question.'''
        return self._allows_duplicate_labels()


    def _allows_duplicate_labels(self):
        '''Subclass may want to overwrite this.'''
        return True


    def _get_supported_dtypes(self):
        '''A subclass is free to overwrite this. Mostly for mdp compatibility.'''
        return [S.float32, S.float64] #todo: Which types to allow?


    def add_logger(self, level=logging.INFO):
        name = self.name
        if name == "" or name is None:
            name = str(self.__class__)
        self.log = logging.getLogger(name)
        self.log.setLevel(level)
        self.log.debug('Adding logger.')


    # Support pickle. We remove the logger
    def __getstate__(self):
        self.log.debug('Removing logger. (__getstate__ called).')
        d = self.__dict__.copy()
        del d['log']
        return d


    def __setstate__(self, d):
        self.__dict__ = d
        self.add_logger()
        self.reset()


    def __str__(self):
        name = self.name
        if not name: name = self.__class__.__name__
        return '<DataSource %s>' % name


    def __repr__(self):
        name = self.name
        if name is None: name = ''
        return '<DataSource %s oudput_dim=%i (%i samples drawn from %s)>' % \
               (name, self.output_dim, self.number_of_samples_until_now, str(self.number_of_samples_max))


    def __call__(self, x=None, n=None, **kargs):
        '''Allow to call with no argument. (mdp does not allow that but
        for a data source it makes sense.)
        @param x:
            This is ignored. It is just passed through to the _execute method.
            Can be None. Usually this is not used by DataSources.
        @param n:
            If given (default None), the number of samples to draw. On default
            the value of self.number_of_samples_execute=1 will be taken.'''
        if x is None:
            x = S.empty((2,0),dtype=self.dtype)
        self.execute(x, n=n, **kargs)


    def __add__(self,other):
        '''
        Adding mdp nodes to a datasource yields a FlowDataSource, which consists
        of a DataSource and a number of mdp.Nodes that are ready to execute.
        '''
        return FlowDataSource( mdp.Node.__add__(self, other) )
    


class FlowDataSource(DataSource):
    '''With a FlowDataSource it is possible to compose a data source that
    consists of any DataSource and a number of mdp.Nodes that act upon 
    each sample from that data source.
    
    For example it is possible to artificially make the data more noise
    by adding a mdp.NoiseNode.

    The __add__ method if DataSource handles the case when you add a Node
    to any DataSource.
    DS_combined = DS + mdp.NoiseNode()

    '''
    def __init__(self, flow, **kws):
        '''Init with an mdp.Flow (where only the first Node can be a DS).'''
        self.flow = flow
        if not isinstance(flow[0],DataSource):
            raise ValueError('The first instance of the mdp.Flow given to a FlowDataSource has to be an instance of DataSource but was %s' % str(type(flow[0])))
        super(FlowDataSource,self).__init__(output_dim=flow[-1].output_dim,
                                            number_of_samples_execute=flow[0].number_of_samples_execute,
                                            number_of_samples_max=flow[0].number_of_samples_max,
                                             **kws)
        
        
    @property
    def allows_duplicate_labels(self):
        return self.flow[0].allows_duplicate_labels


    @property
    def number_of_samples_still_available(self):
        return self.flow[0].number_of_samples_still_available
    
    
    def _get_supported_dtypes(self):
        return self.flow[0]._get_supported_dtypes()
    
    
    def _allows_duplicate_labels(self):
        return self.flow[0]._allows_duplicate_labels()
    
    
    def _get_labels(self, n, start):
        return self.flow[0]._get_labels(n, start)
    
        
    def _reset(self):
        self.flow[0].reset()


    def _samples(self,n=1, **kws):
        d = self.flow[0].samples(n=n, **kws)
        rest = self.flow[1:]
        if len(rest) > 0:
            return rest.execute(d)
        else:   
            return d
    
    
    def __repr__(self):
        name = self.name
        if name is None: name = ''
        return '<FlowDataSource [%s]>' % (', '.join(repr(f) for f in self.flow))

    
    def __str__(self):
        name = self.name
        if name is None: name = ''
        return '<%s>' % ('\n + '.join(str(f) for f in self.flow))

    
    def __add__(self, other):
        self.flow += other
        assert isinstance(other, mdp.Node)
        self._output_dim = self.flow[-1].output_dim
        return self
    
    
    def __getitem__(self,i):
        return self.flow[i]
    
    
    def ranges(self):
        'The ranges of this FlowDataSource is defined by the first DS in the self.flow'
        return self.flow[0].ranges()
    

class SeededDataSource(DataSource):
    '''An abstract DataSource that adds tracking of the random generator's
    state and an optional random seed value.'''
    def __init__(self, seed=None, **kws):
        '''
        @param seed:
            An optional random seed to guarantee, that the data source will
            give the same "random" values as from a previous run.
        @note:
            Subclasses must use self.random instead of scipy.random
            in order to work.
        '''
        super(SeededDataSource,self).__init__(**kws)
        self.seed = seed
        self.random = S.random.RandomState(seed=self.seed)


    def reset(self, seed=None):
        DataSource.reset(self)
        if seed is None:
            self.log.info('Resetting random seed to initial value %s',self.seed)
            seed = self.seed
        self.log.info('Resetting random seed to value %s',str(seed))
        self.random = S.random.RandomState(seed=seed)
        


    def __repr__(self):
        if self.name is None:
            name = ''
        else:
            name = str(self.name)
        return '<DataSource %(name)s oudput_dim=%(output_dim)i with %(n)i/%(max)s samples, seed=%(seed)s>' \
               % dict(name=name, output_dim=self.output_dim, n=self.number_of_samples_until_now, max=str(self.number_of_samples_max), seed=str(self.seed))



class DemoDataSource(SeededDataSource):
    '''A demo of a minimal data source.'''
    def __init__(self, **kws):
        super(DemoDataSource,self).__init__(**kws) # pass forward some args like...


    def _sample(self, **kws):
        # Each datasource can define use args and kws
        s = self.random.random(size=self.output_dim)
        return s



class ProbabilityDataSource(DataSource):
    '''Declares an additional method "probability" and "density".
    ProbabilityDataSource is assumed to be stationary. '''
    def __init__(self,**kws):
        super(ProbabilityDataSource,self).__init__(**kws)


    def probability(self,x):
        '''
        The probability that the point x belongs to this data source.

        Values must be in the half-open interval [0,1) .

        Do not confuse with the probability that a drawn sample == x !
        That would be zero for point-like samples.

        Subclass should implement this'''
        raise NotImplementedError()


    def density(self, shape=None):
        if shape is None:
            shape = tuple([100]*self.output_dim)
        try:
            self._cached_density # just to test if cached
            if self._cached_shape == shape:
                d = self._cached_density
            else:
                raise Exception()
        except:
            shape = S.array(shape)
            d = S.zeros(shape)
            sws = 1.0 / S.array(shape)
            for index in S.ndindex(*shape):
                cx = index * sws
                d[index] = self.probability(cx)
        self._cached_shape = d.shape
        self._cached_density = d
        return d


    def __add__(self,other):
        '''
        Returns a datasource, that is composed of an addition of the probability
        densities (followed by a normalization).
        '''
        return CompositeDataSource([self,other],composition='add',
                                   safemode=self._safemode)

    def __sub__(self, other):
        '''
        Returns a datasource, that is composed of the density of self without
        other. clipTo10(self-other)
        '''
        return CompositeDataSource([self,other],composition='sub',
                                   safemode=self._safemode)


    def __or__(self,other):
        '''
        Returns a datasource, that is composed by max(self,other)
        '''
        return CompositeDataSource([self,other],composition='max',
                                   safemode=self._safemode)


    def __and__(self, other):
        '''
        Returns a datasource, that is composed by min(self,other)
        '''
        return CompositeDataSource([self,other],composition='min',
                                   safemode=self._safemode)


    def __mul__(self, other):
        '''
        Returns a datasource, that is composed of a multiplication of the
        densities of self and other. Where both densities are 1.0 the result
        is 1.0.
        This interpretation is most compatible with probabilistic calculus.
        '''
        return CompositeDataSource([self,other],composition='mul',
                                   safemode=self._safemode)



class DensityDataSource(SeededDataSource,ProbabilityDataSource):
    '''A multi-purpose DataSource with a rasterized density from which you can
    get samples.
    A DensityDataSource has an underlying density from which infinitely many
    samples can be drawn. The format returned is MDP compatible.
    The range of the random samples is 0.0 <= x <= 1.0 in each dimension.
    '''
    def __init__(self, density, sparse=None, **kws):
        '''
        @param density:
            A numpy array with values between 0.0 and 1.0 that represents
            the density to sample from. When sampling, a random element is
            picked and it is compared to a random number that is chosen
            between 0 and 1.
            If that random number is smaller than the entry in the
            density array, the coordinates of that entry are returned after
            being scaled to the interval [0.0, 1.0].
            The density is scaled such that the maximum is 1.0.
            If very few entries with 1.0 are in density, sparse should be True.
        @param sparse:
            This influences how the random samples are drawn. Not the
            probability but the method how to get a random sample is influenced.
            If sparse=None, then a smart strategy is employed, that depends on a
            pre-computed list that stores how probable it is to choose a valid
            sample.
            If sparse=False, then at first a coordinate of a new sample
            is chosen and after that it is checked if a random value is above
            the threshold of the density at the same coordinate.
            (However after self.iterations_when_to_switch_to_sparse
            unsuccessful attempts, it is switched to the sparse-strategy)
            If sparse=True (which is useful for very sparse densities with only
            a few coordinates with higher values), then a random value between
            0 and 1 is drawn first, followed by filtering all coordinates that
            have a higher value that this. Then a random value is chosen from
            the filtered results. The filtering takes quite long for large
            density arrays, but for very sparse densities, it may take longer
            to re-draw the random-values again and again until a coordinate is
            found with a probability high enough.
        '''
        super(DensityDataSource, self).__init__(**kws)
        self._sparse = sparse
        # When to give up sparse=False strategy and switch to sparse=True, even
        # if sparse=False or sparse=None was specified. This is just to avoid
        # endless trying:
        self.iterations_when_to_switch_to_sparse = 2000
        # If sparse=None, then this value decides when to use the one or the
        # other strategy:
        self.probability_when_to_use_sparse = 0.001
        self._min = density.min()
        if self._min < 0.0:
            raise Exception('Density had negative minimum value (%f)' % self._min)
        self._density = density * (1.0 / density.max())
        self._max = self._density.max()
        if self._max < 1.0:
            raise Exception('Maximum of density should be 1.0 but was %f' % self._max)
        self.loadfactors = [ len(where( (i+1.0)/10.0 <= self._density )[0])/float((S.size(self._density)))
                             for i in range(10)]
        #print 'loadfactors', self.loadfactors


    def _sample(self):
        r = self.random.uniform()
        sample = None
        sparse = self._sparse
        if sparse is None:
            # We assume that the "sparse=True"-strategy is 1000times slower
            load = self.loadfactors[int(round(r))]
            # Now load is approximately the probability of getting a valid
            # random sample
            if load < self.probability_when_to_use_sparse:
                sparse = True
            else:
                sparse =False
        if not sparse:
            count = 0
            while not sample:
                count += 1
                coords = [ self.random.randint(low=0, high=self._density.shape[i]-1)
                           for i in range(self.output_dim) ]
                if r < self._density[tuple(coords)]:
                    sample = coords
                if count >= self.iterations_when_to_switch_to_sparse:
                    sparse = True
                    break
        if sparse:
            candidates = where( r < self._density )
            if candidates[0].size == 0:
                raise Exception('Cannot find a random sample for the random value %f' % r)
            lenc = len(candidates[0])
            sample = [ candidates[i][self.random.randint(low=0, high=lenc-1)] for i in range(self.output_dim) ]
        # Now we can take the sample, but we need to map them to the
        # intervals 0..1 for each dimension (and not coordinates of density).
        # And furhter, we jitter them within one cell to avoid grid-effects.
        for i in range(len(sample)):
            sample[i] = ( sample[i] + self.random.uniform() ) / self._density.shape[i]
        return sample


    def probability(self,x):
        '''Unnormalized probability returns 1.0 if point belongs to
        this datasource and 0.0 else.
        @todo !!!'''
        # TODO: find bin in _density x falls in and return that vaule

        raise NotImplementedError()


    def density(self,shape=None):
        if shape is not None:
            raise ValueError("Cannot change shape of density data source.")
        return self._density


    @property
    def output_dim(self):
        return self._density.ndim



class CompositeDataSource(DensityDataSource):
    '''A data source that is composed of two data sources.

    You will rarely use the constructor itself, because the base class
    "ProbabilityDataSource" understands arithmetic operations to create a
    composite data source.

    Example:
       compositeDS = ds1 / ds2  # ds1 without ds2

    @param composition:
        How to combine the (two) data sources.
        Possible options are:
        "add": Just add both densities and normalize. (default)
        "or": Both sources are combined.
        "sub" or synonym "without": All samples from ds1 which are not in ds2
        "mul": Densities are multiplied

    '''
    def __init__(self, datasources=[], weights=None, composition="add",
                 sparse=None, shape=None,
                 **kws):
        assert len(datasources) == 2, 'Datasources to combine must have len==2!'
        assert datasources[0].output_dim == datasources[1].output_dim
        self._output_dim = datasources[0].output_dim
        self.sources = datasources
        self.normedWeights = weights
        self.composition = composition
        d1 = self.sources[0].density(shape=shape)
        d2 = self.sources[1].density(shape=shape)
        if weights is None:
            weights = [d1.sum(),d2.sum()]
            summed = sum(weights)
            if not summed > 0.0: summed = 1.0
            self.normedWeights = [ w/summed for w in weights]
        d = self._build_density(shape)
        super(CompositeDataSource, self).__init__(d,sparse,**kws)
        for s in self.sources:
            assert hasattr(s, 'probability')
            s._safemode = self._safemode
        self._number_of_samples_until_now = 0


    def _build_density(self,shape):
        d1 = self.sources[0].density(shape=shape)
        d2 = self.sources[1].density(shape=shape)
        c = self.composition
        if c == "add":
            d = d1+d2
            d = d.clip(0.0,1.0)
        elif c == "sub":
            d = d1 - d2
            d = d.clip(0.0,1.0)
        elif c == "or":
            d = S.maximum(d1,d2)
        elif c == "and":
            d = S.minimum(d1,d2)
        elif c == "mul":
            d = d1 * d2
        else:
            raise ValueError("Composition " + str(composition) + " not understood.")
        return d


    def density(self, shape=None):
        if shape is not None and self._density.shape != shape:
            # rebuild density if necessary
            self._density = self._build_density(shape)
        return super(CompositeDataSource, self).density(shape)


    def probability(self,x):
        p = 0.0
        p1 = self.sources[0].probability(x)
        p2 =self.sources[1].probability(x)
        if self.composition == "sub":
            p = p1 - p2
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
        elif self.composition == "add":
            p = p1 + p2
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
        elif self.composition == "or":
            p = max(p1,p2)
        elif self.composition == "and":
            p = min(p1,p2)
        elif self.composition == "mul":
            p = p1 * p2
        else:
            raise ValueError("Composition " + str(self.composition) + " not understood.")
        return p


    def _get_supported_dtypes(self):
        inter = set(self.sources[0]._get_supported_dtypes())
        for s in self.sources[1:]:
            inter.intersection_update(set(s._get_supported_dtypes()))
        return list(inter)


    def __getitem__(self,i):
        return self.sources[i]



class CascadedDataSource(DataSource):
    '''A generic wrapper for another data source.

    The trick here that we have to provide the right label information, because
    we may not use the self.source exclusively. Therefore, we get the labels
    right after each sample and cache them locally in self._collected_labels.
    Otherwise we could not provide the functionality to get_labels(n,start).
    
    Caveat: 
        If the labels are really big, this may lead to memory issues.
    '''
    def __init__(self, source=None, **kws ):
        ''''''
        if not isinstance(source, DataSource):
            raise ValueError('CascadedDataSource needs a data source. "source" cannot be empty.')
        super(CascadedDataSource,self).__init__(**kws)
        self._source = source
        self._collected_labels = []
        self._allows_duplicate_labels = self._source._allows_duplicate_labels
        self._get_supported_dtypes = self._source.get_supported_dtypes
        self._number_of_samples_max = self._source.number_of_samples_max
        self.number_of_samples_still_available = self._source.number_of_samples_still_available


    def _sample(self, **kws):
        # Note, when chaining this here, also check NoDuplicatesCascadedDataSource._sample
        s = self._source.sample(**kws)
        self._collected_labels.append( self._source.get_labels(n=1) )
        return s

    
    def _samples(self, n=1, **kws):
        s = self._source.samples(n=n,**kws)
        self._collected_labels.append( self._source.get_labels(n=n) )
        return s
    

    @property
    def number_of_samples_until_now(self):
        return len(self._collected_labels)


    def _get_labels(self, n, start):
        return self._collected_labels[start:start+n]


    def _reset(self):
        self._collected_labels = []
        self._source.reset()


    def __str__(self):
        name = self.name
        if name is None: name=''
        return  "CascadedDataSource "+self.name+" of "+ super(CascadedDataSource,self).__str__()


    def __repr__(self):
        return "<"+self.name+": "+super(CascadedDataSource,self).__repr__()+">"




class NoDuplicatesCascadedDataSource(CascadedDataSource):
    '''Enforces that no duplicate labels can be produced. (Raises DataSourceException)

    Assumes that the cascaded (inner) data source (given to the constructor with
    the keyword "source") implements _get_labels as a list of a list.
    The entries in the inner list are the
    '''
    def __init__(self, **kws):
        super(NoDuplicatesCascadedDataSource, self).__init__(**kws)


    def is_duplicate_label(self,l):
        n = len(l)
        labels = self._collected_labels
        m = len(labels)
        c = 0
        i = 0
        lc = labels[c]
        while c < m:
            if i == n or len(lc)==i:
                return True # reached end of current lc list or end of l (which is i==n)
            if lc[i] != l[i]:
                c += 1 # skip to the next label
                i  = 0 # and begin there from the first item
                lc = labels[c]
                continue
            i += 1
        return False # no duplicate found


    def _sample(self, **kws):
        '''
        @raise DataSourceException:
            If duplicate label information was requested.'''
        s = self.source.sample(**kws)
        l = self.source.get_labels(n=1)
        if self.is_duplicate_label(l):
            raise DataSourceException('No duplicate labels allowed here.')
        self._collected_labels.append( l )
        return s


    def _allows_duplicate_labels(self):
        '''Because we explicitly forbid the same label parameters are used
        more than once.'''
        return False


    def __str__(self):
        name = self.name
        if name is None: name=''
        return  "NoDuplicatesCascadedDataSource "+self.name+" of "+ super(NoDuplicatesCascadedDataSource,self).__str__()


# Two convenience classes
TrainingSetDataSource = NoDuplicatesCascadedDataSource
TestSetDataSource = NoDuplicatesCascadedDataSource
# Use like:
# d0 = ImageDataSource(...)
# TRAIN = TrainingSetDataSource(d0)
# TEST = TestSetDataSource(d0) # from the same underlying d0
# TRAIN.samples(10)
# TEST.samples(10) # will not be the same as the TRAIN samples!
# No image in the test set can be in the train set, now!

#class RepeatingDataSource(CascadedDataSource):
#    '''An infinite data source (a wrapper) which samples from a finite 
#    inner datasource such that all samples are drawn again and again.
#    
#    Example: Let D be a data source which generates [1,2,3], then 
#    RPD = RepeatPermutatedDataSource(D):
#    RPD.samples(10) -> [1,2,3,1,2,3,1,2,3,1]
#    Note that in each three-block each sample is used. 
#    '''
#    def __init__(self, source=None, **kws):
#        if not source.number_of_samples_max < S.Infinity:
#            raise ValueError('RepeatPermutatedDataSource can only be created from a data source that has a finite number of samples (not scipy.Infinity).')
#        super(RepeatingDataSource,self).__init__(source=source,**kws)
#        
#        
#
#class PermutingDataSource(CascadedDataSource):
#    '''A data source that permutes its inner data source.
#    
#    
#    '''
#    def __init__(self, source=None, **kws):
#        if source.number_of_samples_max < S.Infinity:
#            raise ValueError('RepeatPermutatedDataSource can only be created from a data source that has a finite number of samples (not scipy.Infinity).')
#        super(PermutingDataSource,self).__init__(source=source,**kws)
        
        





class ImageDataSource(DataSource):
    '''A generic data source representing rectangular pixel images.
    All images have the same size (of course).

    The generated samples are flat, but can be reshaped with a helper method.
    '''
    def __init__(self, height=100, width=200, channels=3, **kws):
        '''
        @param width:
            The width of the 2D image.
        @param height:
            The height of the image
        @param channels:
            If the image should be in color or monochrome. With or without
            alpha. channels=4 -> RGBA, channels=3-> RGB,
            channels=2->monochrome with alpha, channels=1->monochrome.'''
        super(ImageDataSource, self).__init__(output_dim=width*height*channels, **kws)
        self.height    = height
        self.width     = width
        self.channels  = channels


    @property
    def color(self):
        if self.channels in (3,4): return True
        else: return False


    @property
    def has_alpha(self):
        if self.channels in (2,4): return True
        else: return False


    def _sample(self, **kws):
        raise NotImplemented


    def sample2image(self, sample, channels=-1, fill=0.0):
        '''Reshapes a sample to the correct image shape.
        @param sample.
            A single sample. It is assumed to contain NO alpha values. Only 
            gray or color.
        @param channels:
            The third dimension. Defaults to -1. If you provide a greater
            number if you want to fill the additional values with zeros.
            However, if channels in (2,4) ) the last channel is filled with ones
        '''
        _dim2 = sample.size // (self.width * self.height)
        if channels == -1 or channels == _dim2:
            s = sample.reshape(( self.height, self.width, channels ))
        elif channels > _dim2:
                # we have to add extra values
                self.log.debug('sample2image: Adding ones to the image. channels:%i->%i.', _dim2, channels)
                s = S.zeros((self.height, self.width,channels),dtype=sample.dtype)
                if fill != 0.0:
                    s.fill(fill)
                t = sample.reshape(( self.height, self.width, -1 ))
                t -= t.min() 
                assert t.min() >= 0.0
                t /= t.max()
                s[:,:,:t.shape[2]] = t[:]
                if channels==4 or channels==2:
                    self.log.debug('Quick fix:Setting alpha-channel to 1.0 everywhere because the sample had no alpha at all.')
                    s[:,:,-1] = 1.0
        else:
           raise ValueError("channels cannot be smaller than the intrinsic image, because I don't know which values to drop.")
        return s


    def blanksample(self):
        return S.zeros((1, self.output_dim))


    def quicklook(self, sample,fig=None, name='', **kws):
        import pylab
        pylab.figure(fig)
        pylab.imshow(self.sample2image(sample,channels=3),interpolation='nearest', **kws)
        pylab.title(name)
        pylab.draw()
        print('If you have not already, call pylab.show() once.')



class ImageFilesDataSource(ImageDataSource):
    '''Having a set of image files as the source of objects/views.'''
    def __init__(self, basepath=None, files=[], remove_alpha=True, convert2gray=False, apply=None, **kws):
        '''
        @param basepath: 
            Optionally given path to prepend to all files.
        @param files: 
            An iterable (e.g. a list) with the pathes.
        @param remove_alpha: 
            If True, remove the alpha channel from the images. The background
            will be black (0.0). 
        @param convert2gray: 
            If True (default False). Convert to gray scale image if not already
            
        '''
        self.basepath = basepath
        self._files = files
        self._apply = apply
        self._remove_alpha = remove_alpha
        self._convert2gray = convert2gray
        # open the first image
        self._safemode = False # just temporary, will be overwritten from super.__init__
        f = self._imread(iter(files).next())
        height   = f.shape[0]
        width    = f.shape[1]
        if f.ndim == 2:
            channels = 1 # gray
        else:
            channels = f.shape[2]
        super(ImageFilesDataSource, self).__init__(height=height,
                                                   width=width,
                                                   channels=channels,
                                                   #number_of_samples_max = len(files),
                                                   **kws)
    
    @property
    def files(self):
        return self._files
        
        
    def _imread(self, path):
        '''Private helper to read an image and convert to numpy array.'''
        if self.basepath:
            path = self.basepath + os.sep + path 
        try:
            from PIL import Image
            arr = S.array(Image.open(path),dtype=S.float32) / 255.
        except ImportError, e:
            if path.lower().endswith('.png'):
                try:
                    from matplotlib import pyplot as PLT
                    arr = PLT.imread(path) # this is automatically a float betw. 0 and 1
                except ImportError as e:
                    self.log.exception('PIL and matplotlib not available. Cannot open image %s' %path)
            else:
                raise ImportError('Cannot import ' + path + ' file type.')
        if self._remove_alpha and arr.shape[2] in (2,4):
            arr = arr[:,:,:-1] * arr[:,:,-1:] # to avoid shared edges, we apply the alpha as a mask
        if self._convert2gray:
            if arr.shape[2] == 3: # if color
                arr = arr.sum(axis=2) 
                arr /= 3.
            elif arr.shape[2] == 4:
                arr = arr[:,:,0:3].sum(axis=2)
                arr /= 3.
        if self._safemode:
            if self.channels == 1:
                assert arr.shape == (self.height, self.width), 'Image %s has not the expected shape. %s but expected (%i,%i).' %(path, str(arr.shape), self.height, self.width)
            else:
                assert arr.shape == (self.height, self.width, self.channels), 'Image %s has not the expected shape %s but expected (%i,%i,%i).' %(path, str(arr.shape), self.height, self.width, self.channels)
        return arr

            
    def _sample(self, nr=None, **kws):
        if not nr:
            nr = self.number_of_samples_until_now
            self._number_of_samples_until_now += 1
        arr = self._imread(self.files[nr]).flatten()
        return arr 
        
        

class ImageFilesObjViewDataSource(ImageFilesDataSource, SeededDataSource):
    '''Describing a set of image files as objXX__YY.EXT. 
    
    There are many different naming schemes possible, as long as they can be
    described by an template and iterators to fill the numbers of the template.
    It's only important that each possible combination of the variable are 
    valid. For example if obj=1 has view=100, then the other objects should
    have a view=100, too.'''
    def __init__(self, 
                 basepath=None,
                 filename_template="obj%(obj)i__%(view)i.png", 
                 iterators_dict=dict(obj=[1],view=[0]), **kws):
        '''
        @param filename_tmplate: 
            A format string that is filled with the iterators defined 
            by the iterators_dict.
        @param iterators_dict: 
            A dict with iterators over ints. The keys of the dict must match the 
            names given in the format string.
            The dict can contain a lists, xrange object and so forth. They must
            support __len__(), __getitem__(i) and the integer values must end
            with the largest value. 
             
        Example: two objects with 100 views each
            DS = ImageFilesObjViewDataSource(
                    iterators_dict=dict(obj=[1,2], view=xrange(100) ) )    
            '''
        self.filename_template = filename_template
        
        self.iterators_dict = iterators_dict
        self._labels_order = iterators_dict.keys()
        self._labels = []
        # get a first image. We load this to infer the size and check that 
        # the path is okay.
        #d = dict()
        #m = 1
        #for k, v in iterators_dict.items():
            #m *= len(v)
            #d[k] = v[0] # get the first element from the iterator v 
        first = filename_template % dict((k,v[0]) for k, v in iterators_dict.items())
        super(ImageFilesObjViewDataSource, self).__init__(basepath=basepath,
                                                          files=[first],
                                                          **kws)
            
        # we did not provide a complete list of files to the ImageFilesDataSource,
        # (it might be to many), so now we have to assign the correct size: 
        #self._number_of_samples_max = m 
    
    @property
    def files(self):
        import loop_unroller
        r = loop_unroller.Unroller(loops=self.iterators_dict.items())
        for ns, vs in r:
            yield self.filename_template % dict(zip(ns,vs))
    
    
    def _sample(self, **kws):
        d = dict()
        special_request = False 
        for k,v in self.iterators_dict.items():
            if k in kws.keys():
                d[k] = kws[k]
                special_request = True
            else:
                r = None
                while r not in v:
                    # Perhaps we have to draw several times, if the range
                    # v had a stepsize other than 1.
                    r = self.random.randint(low=v[0], high=v[-1]+1)
                d[k] = r
        if not special_request: self._number_of_samples_until_now += 1
        self._labels.append(list(d[l] for l in self._labels_order))
        arr = self._imread(self.filename_template % d).flatten()
        return arr
        
        
    def _get_labels(self, n, start):
        return self._labels[start:start+n]



class UniformDataSource(SeededDataSource, ProbabilityDataSource):
    def __init__(self,
                 limits=[ [0.0, 1.0], [0.0, 1.0] ],
                 **kws):
        ''''''
        self.limits=limits
        super(UniformDataSource, self).__init__(**kws)
        assert len(limits) == self.output_dim


    def _sample(self):
        r = self.random.uniform(self.output_dim)
        x = S.zeros(self.output_dim)
        for d in range(self.ouput_dim):
            x[d] = r[d] * (self.limits[d][1] - self.limits[d][0]) + \
                   self.limits[d][0]
        return x


    def probability(self,x):
        '''Unnormalized probability returns 1.0 if point belongs to
        this data source and 0.0 else.'''
        # Faster implementation than the density based approach
        for d in range(self.output_dim):
            if self.limits[d][0] <= x[d] <= self.limits[d][1]:
                continue
            else:
                return 0.0
        return 1.0



class NoisyFigure3dDataSource(SeededDataSource):
    '''
    Base class for 3D figures (of any kind) that can be rotated, scaled and
    translation (moved).

    A class that inherits from this one should provide a _samples(n) method
    to generate any 3d data points in the cube from [0,1] in each dimension
    (x,y,z).

    Make sure that due to the scale and translation parameter, the data points
    are _not_ moved out of the cube [0,0,0] to [1,1,1].
    However, if clip=true there is a mechanism that checks if points lie
    outside of this cube and request a new _samples(1) for these cases.
    So you do not have to care if due to noise, sometimes a point leaves the
    valid cube.

    The noise that is added is a kind of global and the same for all data
    points. If you want noise that is intrinsic to the figure, you must
    implement it in your subclass.
    '''
    def __init__(self,
                 phi=0.0,
                 theta=0.0,
                 psi=0.0,
                 translation=[0.,0.,0.],
                 scale=[1.,1.,1.],
                 global_noisy=False,
                 global_noise_func=S.random.RandomState().normal,
                 global_noise_args=(0,.1),
                 global_noise_type='additive',
                 clip=True,
                 **kws):
        '''Create a new Figure 3D.

        Applies A rotation around phi, theta and psi to each data point around
        the position [0.5, 0.5, 0.5].
        Then a translation and scale i applied.

        After that optional nose is added.

        If global_noisy=True, the global_noise_func, global_noise_args and
        global_noise_type are forwarded to a mdp.NoiseNode to add that noise
        _after_ the rotation, translation and scale has been applied..


        @param clip:
            If true (default) the samples are clipped to the unit-cube.
        '''
        super(NoisyFigure3dDataSource, self).__init__(output_dim=3, **kws)
        # Build up the rotation matrices:
        rot_phi          = S.diag([1., S.cos(phi), S.cos(phi)])
        rot_phi[1,2]     = -S.sin(phi)
        rot_phi[2,1]     = S.sin(phi)
        rot_theta        = S.diag([S.cos(theta), 1., S.cos(theta)])
        rot_theta[0,2]   = S.sin(theta)
        rot_theta[2,0]   = -S.sin(theta)
        rot_psi          = S.diag([S.cos(psi), S.cos(psi), 1.])
        rot_psi[0,1]     = -S.sin(psi)
        rot_psi[1,0]     = S.sin(psi)

        self.rot         = S.dot(S.dot(rot_phi, rot_theta), rot_psi)

        self.translation = S.array(translation, dtype=S.float64)

        self.scale       = S.array(scale, dtype=S.float64)

        self.global_noisy = global_noisy
        if self.global_noisy:
            self.global_noise = mdp.nodes.NoiseNode(
                                         noise_func=global_noise_func,
                                         noise_args=global_noise_args,
                                         noise_type=global_noise_type)


    def _sample(self, **kws):
        return self.samples(1, **kws)[0]


    def _samples(self,n, **kws):
        s = super(NoisyFigure3dDataSource, self)._samples(n, **kws)
        # We rotate around the point [ .5, .5, .5]:
        s -= S.array([.5,.5,.5])
        s *= self.scale
        s = S.dot(s, self.rot)
        s += S.array([.5,.5,.5]) + self.translation

        if self.global_noisy:
            s = self.global_noise(s)
            # Get the indices of the ids that are out of the 0.0..1.0 range:
        if self.clip:
            bad = S.unique( S.where( (s < 0.0) + (s > 1.0) )[0] )
            for i in bad:
                s[i] = self._samples(1)[0] # _sample() always returns valid vals
        return s


    def density(self):
        '''Just a place holder for density.'''
        raise NotImplementedError()


